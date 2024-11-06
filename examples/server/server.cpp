#include "common.h"

#include "whisper.h"

#include "httplib.h"

#include "json.hpp"

#include <atomic>

#include <chrono>

#include <condition_variable>

#include <csignal>

#include <fstream>

#include <future>

#include <iomanip>

#include <memory>

#include <mutex>

#include <queue>

#include <sstream>

#include <string>

#include <thread>

#include <vector>

#include "thread_pool.h"

#include <future>

#include <map>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif


// Constants for configuration
const size_t MAX_UPLOAD_SIZE = 10 * 1024 * 1024; // 10 MB max upload size

namespace {
  using json = nlohmann::ordered_json;
}

namespace whisper_server {

    // Specialization for arrays with known size
    template<typename T>
    typename std::enable_if<std::is_array<T>::value && std::extent<T>::value != 0,
                          std::unique_ptr<T>>::type
    make_unique(std::size_t) = delete;

    // Specialization for arrays with unknown size
    template<typename T>
    typename std::enable_if<std::is_array<T>::value && std::extent<T>::value == 0,
                          std::unique_ptr<T>>::type
    make_unique(std::size_t n) {
        typedef typename std::remove_extent<T>::type U;
        return std::unique_ptr<T>(new U[n]());
    }

  const int PROCESSING_TIMEOUT_MS = 30000; // Max time for processing audio

  struct server_params {
    std::string hostname = "127.0.0.1";
    std::string public_path = "examples/server/public";
    std::string request_path = "";
    std::string inference_path = "/inference";
    int32_t n_threads_http = std::min(4, static_cast < int32_t > (std::thread::hardware_concurrency()));

    int32_t port = 8080;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
    bool ffmpeg_converter = false;
  };

  struct whisper_params {
    int32_t n_threads = std::min(4, static_cast < int32_t > (std::thread::hardware_concurrency()));
    // These are "Whisper" processors. It's a strategy that was explored to split the input audio into -p chunks and process the chunks in parallel. Sometimes it can speed-up the process, but the transcription result at the splits will likely be worse due to partial words and losing context.
    // A rule of thumb is that -p * -t should be less or equal to the number of CPU cores that you have.
    int32_t n_processors = 1;
    // number of whisper instances to run in parallel != processors
    int32_t n_instances = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = 2;
    int32_t beam_size = -1;
    int32_t audio_ctx = 0;
    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    float temperature = 0.00f;
    float temperature_inc = 0.20f;
    bool translate = false;
    bool detect_language = false;
    bool diarize = false;
    bool tinydiarize = false;
    bool split_on_word = false;
    bool no_timestamps = false;
    bool use_gpu = true;
    bool flash_attn = false;
    bool debug_mode = false;
    bool no_fallback = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_realtime = false;
    bool print_progress = false;
    std::string language = "en";
    std::string prompt = "";
    std::string model = "models/ggml-base.en.bin";
    std::string response_format = "json";
    std::string openvino_encode_device = "CPU";
    std::string dtw = "";
    std::string font_path = "";
  };

  // Cross-platform signal handling
  std::atomic < bool > exit_flag {
    false
  };

  // Function to parse string to bool
  static bool parse_str_to_bool(const std::string & value) {
    std::string lower_value = value;
    std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
    return (lower_value == "true" || lower_value == "1");
  }

  static void signal_handler(int /*signal*/ ) {
    exit_flag.store(true);
  }

  // Calculate the number of instances based on available GPU memory
  static int calculate_num_instances(const whisper_params &params ) {
   // return num_instances
   return params.n_instances;
  }

  static std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto itt = std::chrono::system_clock::to_time_t(now);
    auto ms =
      std::chrono::duration_cast < std::chrono::milliseconds > (now.time_since_epoch()) % 1000;
    std::ostringstream ss;
    ss << std::put_time(std::localtime( & itt), "%Y-%m-%d %H:%M:%S") << '.' <<
      std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
  }

  // Encapsulate whisper_context in a RAII class
  class WhisperContext {
    public: WhisperContext(const std::string & model_path,
        const whisper_context_params & params) {
        ctx = whisper_init_from_file_with_params(model_path.c_str(), params);
        if (!ctx) {
          throw std::runtime_error("Failed to initialize whisper context");
        }
      }

      ~WhisperContext() {
        if (ctx) {
          whisper_free(ctx);
        }
      }

    // Disable copy
    WhisperContext(const WhisperContext & ) = delete;
    WhisperContext & operator = (const WhisperContext & ) = delete;

    // Allow move
    WhisperContext(WhisperContext && other) noexcept: ctx(other.ctx) {
      other.ctx = nullptr;
    }

    WhisperContext & operator = (WhisperContext && other) noexcept {
      if (this != & other) {
        if (ctx) {
          whisper_free(ctx);
        }
        ctx = other.ctx;
        other.ctx = nullptr;
      }
      return * this;
    }

    whisper_context * get() const {
      return ctx;
    }

    private: whisper_context * ctx;
  };

  struct whisper_instance {
    int id;
    std::shared_ptr < WhisperContext > ctx;
    std::atomic < bool > in_use;
    std::chrono::steady_clock::time_point last_use;

    whisper_instance(int id, std::shared_ptr < WhisperContext > c): id(id), ctx(std::move(c)), in_use(false) {}
  };
  struct InstanceAvailablePred {
          bool operator()(const std::shared_ptr<whisper_instance>& inst) const {
              return !inst->in_use.load(std::memory_order_relaxed);
          }
      };

      class whisper_pool {
      public:
          whisper_pool(const whisper_params& params, int num_instances) {
              std::cout << "[" << current_timestamp() << "] Initializing whisper pool with "
                        << num_instances << " instances\n";

              for (int i = 0; i < num_instances; ++i) {
                  try {
                      auto instance = init_whisper(params, i);
                      instances.emplace_back(std::move(instance));
                  } catch (const std::exception& e) {
                      std::cerr << "[" << current_timestamp() << "] Failed to initialize instance "
                                << i << ": " << e.what() << "\n";
                  }
              }

              if (instances.empty()) {
                  throw std::runtime_error("Failed to initialize any whisper instances");
              }

              // Store params for potential reinitialization
              this->params = params;

              std::cout << "[" << current_timestamp() << "] Successfully initialized "
                        << instances.size() << " instances\n";
          }

          ~whisper_pool() {
              shutdown();
          }

          std::shared_ptr<whisper_instance> get_instance() {
              size_t attempts = 0;
              const size_t max_attempts = instances.size();

              while (attempts++ < max_attempts && !exit_flag.load(std::memory_order_relaxed)) {
                  size_t current = next_instance.fetch_add(1, std::memory_order_relaxed) % instances.size();

                  auto& inst = instances[current];
                  bool expected = false;
                  if (inst->in_use.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
                      inst->last_use = std::chrono::steady_clock::now();
                      return inst;
                  }
              }

              std::unique_lock<std::mutex> lock(mutex);

              auto pred = [this]() -> bool {
                  return std::any_of(instances.begin(), instances.end(), InstanceAvailablePred()) ||
                         exit_flag.load(std::memory_order_relaxed);
              };

              if (condition.wait_for(lock, std::chrono::milliseconds(INSTANCE_TIMEOUT_MS), pred)) {
                  for (size_t i = 0; i < instances.size(); ++i) {
                      size_t index = (next_instance.fetch_add(1, std::memory_order_relaxed)) % instances.size();
                      auto& inst = instances[index];
                      bool expected = false;
                      if (inst->in_use.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
                          inst->last_use = std::chrono::steady_clock::now();
                          return inst;
                      }
                  }
              }

              return nullptr;
          }

          void release_instance(std::shared_ptr<whisper_instance> instance) {
              if (!instance) return;

              instance->in_use.store(false, std::memory_order_release);

              // Notify waiting threads
              {
                  std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
                  if (lock) condition.notify_one();
              }
          }

          void shutdown() {
              exit_flag.store(true, std::memory_order_release);
              condition.notify_all();

              // Force release all instances
              for (auto& instance : instances) {
                  if (instance->in_use.load(std::memory_order_relaxed)) {
                      release_instance(instance);
                  }
              }
          }


          size_t get_instance_count() const {
              return instances.size();
          }

      private:
          std::vector<std::shared_ptr<whisper_instance>> instances;
          std::mutex mutex;
          std::condition_variable condition;
          std::atomic<size_t> next_instance{0};
          std::atomic<bool> exit_flag{false};
          whisper_params params;  // Store for reinitialization

          static constexpr int INSTANCE_TIMEOUT_MS = 30000;  // Max time to wait for an instance

          std::shared_ptr<whisper_instance> init_whisper(const whisper_params& params, int id) {
              whisper_context_params cparams = whisper_context_default_params();
              cparams.use_gpu = params.use_gpu;
              cparams.flash_attn = params.flash_attn;

              auto ctx = std::make_shared<WhisperContext>(params.model, cparams);
              return std::make_shared<whisper_instance>(id, ctx);
          }
      };

  // Function to print usage/help information
  static void whisper_print_usage(int /*argc*/ , char ** argv,
    const whisper_params & params,
      const server_params & sparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] \n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -i N,      --instances N       [%-7d] number of whisper instances to use\n", params.n_instances);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n", params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n", params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n", params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -ml N,     --max-len N         [%-7d] maximum segment length in characters\n", params.max_len);
    fprintf(stderr, "  -sow,      --split-on-word     [%-7s] split on word rather than on token\n", params.split_on_word ? "true" : "false");
    fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n", params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n", params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N       [%-7d] audio context size (0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -wt N,     --word-thold N      [%-7.2f] word timestamp probability threshold\n", params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for decoder fail\n", params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold for decoder fail\n", params.logprob_thold);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n", params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -tr,       --translate         [%-7s] translate from source language to english\n", params.translate ? "true" : "false");
    fprintf(stderr, "  -di,       --diarize           [%-7s] stereo audio diarization\n", params.diarize ? "true" : "false");
    fprintf(stderr, "  -tdrz,     --tinydiarize       [%-7s] enable tinydiarize (requires a tdrz model)\n", params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -nf,       --no-fallback       [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,       --print-special     [%-7s] print special tokens\n", params.print_special ? "true" : "false");
    fprintf(stderr, "  -pc,       --print-colors      [%-7s] print colors\n", params.print_colors ? "true" : "false");
    fprintf(stderr, "  -pr,       --print-realtime    [%-7s] print output in realtime\n", params.print_realtime ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n", params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n", params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect)\n", params.language.c_str());
    fprintf(stderr, "  -dl,       --detect-language   [%-7s] exit after automatically detecting language\n", params.detect_language ? "true" : "false");
    fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt\n", params.prompt.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n", params.openvino_encode_device.c_str());
    // server params
    fprintf(stderr, "  -dtw MODEL --dtw MODEL         [%-7s] compute token-level timestamps\n", params.dtw.c_str());
    fprintf(stderr, "  --host HOST,                   [%-7s] Hostname/ip-address for the server\n", sparams.hostname.c_str());
    fprintf(stderr, "  --ht N,    --http-threads N     [%-7d] number of threads to use for HTTP server\n", sparams.n_threads_http);
    fprintf(stderr, "  --port PORT,                   [%-7d] Port number for the server\n", sparams.port);
    fprintf(stderr, "  --public PATH,                 [%-7s] Path to the public folder\n", sparams.public_path.c_str());
    fprintf(stderr, "  --request-path PATH,           [%-7s] Request path for all requests\n", sparams.request_path.c_str());
    fprintf(stderr, "  --inference-path PATH,         [%-7s] Inference path for all requests\n", sparams.inference_path.c_str());
    fprintf(stderr, "  --convert,                     [%-7s] Convert audio to WAV, requires ffmpeg on the server\n", sparams.ffmpeg_converter ? "true" : "false");
    fprintf(stderr, "\n");
  }

  // Function to parse request parameters
  static void get_req_parameters(const httplib::Request & req, whisper_params & params) {
    if (req.has_param("offset_t")) {
      params.offset_t_ms = std::stoi(req.get_param_value("offset_t"));
    }
    if (req.has_param("offset_n")) {
      params.offset_n = std::stoi(req.get_param_value("offset_n"));
    }
    if (req.has_param("duration")) {
      params.duration_ms = std::stoi(req.get_param_value("duration"));
    }
    if (req.has_param("max_context")) {
      params.max_context = std::stoi(req.get_param_value("max_context"));
    }
    if (req.has_param("max_len")) {
      params.max_len = std::stoi(req.get_param_value("max_len"));
    }
    if (req.has_param("best_of")) {
      params.best_of = std::stoi(req.get_param_value("best_of"));
    }
    if (req.has_param("beam_size")) {
      params.beam_size = std::stoi(req.get_param_value("beam_size"));
    }
    if (req.has_param("audio_ctx")) {
      params.audio_ctx = std::stoi(req.get_param_value("audio_ctx"));
    }
    if (req.has_param("word_thold")) {
      params.word_thold = std::stof(req.get_param_value("word_thold"));
    }
    if (req.has_param("entropy_thold")) {
      params.entropy_thold = std::stof(req.get_param_value("entropy_thold"));
    }
    if (req.has_param("logprob_thold")) {
      params.logprob_thold = std::stof(req.get_param_value("logprob_thold"));
    }
    if (req.has_param("debug_mode")) {
      params.debug_mode = parse_str_to_bool(req.get_param_value("debug_mode"));
    }
    if (req.has_param("translate")) {
      params.translate = parse_str_to_bool(req.get_param_value("translate"));
    }
    if (req.has_param("diarize")) {
      params.diarize = parse_str_to_bool(req.get_param_value("diarize"));
    }
    if (req.has_param("tinydiarize")) {
      params.tinydiarize = parse_str_to_bool(req.get_param_value("tinydiarize"));
    }
    if (req.has_param("split_on_word")) {
      params.split_on_word = parse_str_to_bool(req.get_param_value("split_on_word"));
    }
    if (req.has_param("no_timestamps")) {
      params.no_timestamps = parse_str_to_bool(req.get_param_value("no_timestamps"));
    }
    if (req.has_param("language")) {
      params.language = req.get_param_value("language");
    }
    if (req.has_param("detect_language")) {
      params.detect_language = parse_str_to_bool(req.get_param_value("detect_language"));
    }
    if (req.has_param("prompt")) {
      params.prompt = req.get_param_value("prompt");
    }
    if (req.has_param("response_format")) {
      params.response_format = req.get_param_value("response_format");
    }
    if (req.has_param("temperature")) {
      params.temperature = std::stof(req.get_param_value("temperature"));
    }
    if (req.has_param("temperature_inc")) {
      params.temperature_inc = std::stof(req.get_param_value("temperature_inc"));
    }
  }

  // Updated parse_params function
  static bool parse_params(int argc, char ** argv, whisper_params & params, server_params & sparams) {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "-h" || arg == "--help") {
        whisper_print_usage(argc, argv, params, sparams);
        exit(0);
      } else if (arg == "-t" || arg == "--threads") {
        params.n_threads = std::stoi(argv[++i]);
      } else if (arg == "-p" || arg == "--processors") {
        params.n_processors = std::stoi(argv[++i]);
      } else if (arg == "-i" || arg == "--instances") {
        params.n_instances = std::stoi(argv[++i]);
      } else if (arg == "-ot" || arg == "--offset-t") {
        params.offset_t_ms = std::stoi(argv[++i]);
      } else if (arg == "-on" || arg == "--offset-n") {
        params.offset_n = std::stoi(argv[++i]);
      } else if (arg == "-d" || arg == "--duration") {
        params.duration_ms = std::stoi(argv[++i]);
      } else if (arg == "-mc" || arg == "--max-context") {
        params.max_context = std::stoi(argv[++i]);
      } else if (arg == "-ml" || arg == "--max-len") {
        params.max_len = std::stoi(argv[++i]);
      } else if (arg == "-bo" || arg == "--best-of") {
        params.best_of = std::stoi(argv[++i]);
      } else if (arg == "-bs" || arg == "--beam-size") {
        params.beam_size = std::stoi(argv[++i]);
      } else if (arg == "-ac" || arg == "--audio-ctx") {
        params.audio_ctx = std::stoi(argv[++i]);
      } else if (arg == "-wt" || arg == "--word-thold") {
        params.word_thold = std::stof(argv[++i]);
      } else if (arg == "-et" || arg == "--entropy-thold") {
        params.entropy_thold = std::stof(argv[++i]);
      } else if (arg == "-lpt" || arg == "--logprob-thold") {
        params.logprob_thold = std::stof(argv[++i]);
      } else if (arg == "-debug" || arg == "--debug-mode") {
        params.debug_mode = true;
      } else if (arg == "-tr" || arg == "--translate") {
        params.translate = true;
      } else if (arg == "-di" || arg == "--diarize") {
        params.diarize = true;
      } else if (arg == "-tdrz" || arg == "--tinydiarize") {
        params.tinydiarize = true;
      } else if (arg == "-sow" || arg == "--split-on-word") {
        params.split_on_word = true;
      } else if (arg == "-nf" || arg == "--no-fallback") {
        params.no_fallback = true;
      } else if (arg == "-fp" || arg == "--font-path") {
        params.font_path = argv[++i];
      } else if (arg == "-ps" || arg == "--print-special") {
        params.print_special = true;
      } else if (arg == "-pc" || arg == "--print-colors") {
        params.print_colors = true;
      } else if (arg == "-pr" || arg == "--print-realtime") {
        params.print_realtime = true;
      } else if (arg == "-pp" || arg == "--print-progress") {
        params.print_progress = true;
      } else if (arg == "-nt" || arg == "--no-timestamps") {
        params.no_timestamps = true;
      } else if (arg == "-l" || arg == "--language") {
        params.language = argv[++i];
      } else if (arg == "-dl" || arg == "--detect-language") {
        params.detect_language = true;
      } else if (arg == "--prompt") {
        params.prompt = argv[++i];
      } else if (arg == "-m" || arg == "--model") {
        params.model = argv[++i];
      } else if (arg == "-oved" || arg == "--ov-e-device") {
        params.openvino_encode_device = argv[++i];
      } else if (arg == "-dtw" || arg == "--dtw") {
        params.dtw = argv[++i];
      } else if (arg == "-ng" || arg == "--no-gpu") {
        params.use_gpu = false;
      } else if (arg == "-fa" || arg == "--flash-attn") {
        params.flash_attn = true;
      }
      // server params
      else if (arg == "--port") {
        sparams.port = std::stoi(argv[++i]);
      } else if (arg == "--host") {
        sparams.hostname = argv[++i];
      } else if (arg == "--public") {
        sparams.public_path = argv[++i];
      } else if (arg == "--request-path") {
        sparams.request_path = argv[++i];
      } else if (arg == "--inference-path") {
        sparams.inference_path = argv[++i];
      } else if (arg == "--http-threads" || arg == "-ht") {
        sparams.n_threads_http = std::stoi(argv[++i]);
      }
      else if (arg == "--convert") {
        sparams.ffmpeg_converter = true;
      } else {
        fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
        whisper_print_usage(argc, argv, params, sparams);
        exit(0);
      }
    }

    return true;
  }

  static std::string process_audio(
      whisper_context* ctx,
      const whisper_params& params,
      const std::vector<float>& pcmf32,
      const std::vector<std::vector<float>>& pcmf32s) {

      whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
      wparams.print_realtime = false;
      wparams.print_progress = false;
      wparams.print_timestamps = !params.no_timestamps;
      wparams.translate = params.translate;
      wparams.language = params.language.c_str();
      wparams.n_threads = params.n_threads;
      wparams.offset_ms = params.offset_t_ms;
      wparams.duration_ms = params.duration_ms;

      if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(),
          params.n_processors) != 0) {
          throw std::runtime_error("Failed to process audio");
      }

      if (params.response_format == "json") {
          json result = {
              {"text", whisper_full_get_segment_text(ctx, 0)}
          };
          return result.dump();
      } else {
          return whisper_full_get_segment_text(ctx, 0);
      }
  }

} // namespace whisper_server
int main(int argc, char ** argv) {
  using namespace whisper_server; // Limit scope

  whisper_params params;
  server_params sparams;

  if (!parse_params(argc, argv, params, sparams)) {
    return 1;
  }

  // Cross-platform signal handling
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  int num_instances = calculate_num_instances(params);
  auto pool = make_unique < whisper_pool > (params, num_instances);

  httplib::Server svr;
  // Set number of threads for request handling
  svr.new_task_queue = [&sparams] { return new httplib::ThreadPool(sparams.n_threads_http); };
  svr.set_default_headers({
    {
      "Server",
      "whisper.cpp"
    },
    {
      "Access-Control-Allow-Origin",
      "*"
    },
    {
      "Access-Control-Allow-Headers",
      "content-type, authorization"
    },
  });

  int num_threads = num_instances;
  // Create a thread pool with a suitable number of threads
  ThreadPool thread_pool(num_threads);

  // Set maximum request size (Use set_payload_max_length)
  svr.set_payload_max_length(MAX_UPLOAD_SIZE);

  // POST /inference handler
  svr.Post(sparams.request_path + sparams.inference_path,
          [&](const httplib::Request& req_, httplib::Response& res_) {
              const auto request_start = std::chrono::steady_clock::now();

              if (!req_.has_file("file")) {
                  res_.status = 400;
                  res_.set_content(R"({"error":"Missing 'file' in request"})", "application/json");
                  return;
              }

              const auto audio_file = req_.get_file_value("file");
              if (audio_file.content.size() > MAX_UPLOAD_SIZE) {
                  res_.status = 413;
                  res_.set_content(R"({"error":"File too large"})", "application/json");
                  return;
              }

              std::vector<float> pcmf32;
              std::vector<std::vector<float>> pcmf32s;
              if (!::read_wav(audio_file.content, pcmf32, pcmf32s, params.diarize)) {
                  res_.status = 400;
                  res_.set_content(R"({"error":"Failed to read audio"})", "application/json");
                  return;
              }

              std::promise<json> result_promise;
              std::future<json> result_future = result_promise.get_future();

              const auto queue_start = std::chrono::steady_clock::now();

              // Create shared pointers for moved values
              auto p_pcmf32 = std::make_shared<std::vector<float>>(std::move(pcmf32));
              auto p_pcmf32s = std::make_shared<std::vector<std::vector<float>>>(std::move(pcmf32s));
              auto p_promise = std::make_shared<std::promise<json>>(std::move(result_promise));

              thread_pool.enqueue([queue_start, request_start, p_pcmf32, p_pcmf32s, &pool, &params, p_promise]() {
                  try {
                      auto instance = pool->get_instance();
                      if (!instance) {
                          json error_response = {
                              {"error", "No available instances"},
                              {"status", 503}
                          };
                          p_promise->set_value(error_response);
                          return;
                      }
                      auto process_result = process_audio(instance->ctx->get(), params, *p_pcmf32, *p_pcmf32s);

                      pool->release_instance(instance);


                      json response;
                      if (params.response_format == "json") {
                          response = json::parse(process_result);
                      } else {
                          response["text"] = process_result;
                      }

                      response["status"] = 200;

                      p_promise->set_value(response);

                  } catch (const std::exception& e) {
                      p_promise->set_value({
                          {"error", "Internal server error"},
                          {"status", 500}
                      });
                  }
              });

              if (result_future.wait_for(std::chrono::milliseconds(PROCESSING_TIMEOUT_MS))
                  == std::future_status::timeout) {
                  res_.status = 504;
                  res_.set_content(R"({"error":"Processing timeout"})", "application/json");
                  return;
              }

              json result = result_future.get();
              int status = result["status"].get<int>();
              result.erase("status");

              res_.status = status;
              res_.set_content(result.dump(), "application/json");
          });

  // Error handler
  svr.set_error_handler([](const httplib::Request & /*req*/ , httplib::Response & res) {
    res.status = 404; // Not Found
    res.set_content("{\"error\":\"Invalid request\"}", "application/json");
  });

  svr.set_read_timeout(sparams.read_timeout);
  svr.set_write_timeout(sparams.write_timeout);

  if (!svr.bind_to_port(sparams.hostname.c_str(), sparams.port)) {
    std::cerr << "[" << current_timestamp() <<
      "] Couldn't bind to server socket: hostname=" << sparams.hostname <<
      " port=" << sparams.port << "\n";
    return 1;
  }

  // Restrict served files
  svr.set_mount_point("/", sparams.public_path.c_str());

  //print threds
  std::cout << "Number of threads: " << num_threads << std::endl;
  std::cout << "[" << current_timestamp() << "] Whisper server listening at http://" <<
    sparams.hostname << ":" << sparams.port << " with " << num_instances <<
    " model instances";

  // Start server in a separate thread
  std::thread server_thread([ & ]() {
    if (!svr.listen_after_bind()) {
      std::cerr << "[" << current_timestamp() << "] Error starting server\n";
      exit_flag.store(true);
    }
  });

  // Wait for exit signal
  while (!exit_flag.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Server shutdown
  svr.stop();
  pool -> shutdown();
  thread_pool.shutdown();

  // Join the server thread to ensure it has finished
  if (server_thread.joinable()) {
    server_thread.join();
  }

  std::cout << "[" << current_timestamp() << "] Server shut down gracefully.\n";

  return 0;
}
