#pragma once

#include "whisper.h"
#include "whisper_params.h"
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>
#include <queue>
using namespace whisper;

// RAII wrapper for whisper_context
class WhisperContext {
public:
    WhisperContext(const std::string& model_path, const whisper_context_params& params) {
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

    whisper_context* get() { return ctx; }

    // Prevent copying
    WhisperContext(const WhisperContext&) = delete;
    WhisperContext& operator=(const WhisperContext&) = delete;

private:
    whisper_context* ctx;
};

// Thread-safe pool of whisper contexts using lock-free queue
class WhisperContextPool {
public:
    struct Instance {
        whisper_context* ctx;
        whisper::params params;
        whisper::params default_params;
    };

private:
    struct PaddedContext {
        std::unique_ptr<WhisperContext> context;
        uint32_t core_affinity;
        char padding[32];

        PaddedContext() = default;
        PaddedContext(PaddedContext&& other) noexcept 
            : context(std::move(other.context))
            , core_affinity(other.core_affinity) {}

        PaddedContext& operator=(PaddedContext&& other) noexcept {
            if (this != &other) {
                context = std::move(other.context);
                core_affinity = other.core_affinity;
            }
            return *this;
        }

        PaddedContext(const PaddedContext&) = delete;
        PaddedContext& operator=(const PaddedContext&) = delete;
    };

    std::vector<PaddedContext> contexts;
    std::vector<size_t> in_use_indices;
    std::unique_ptr<LockFreeQueue<size_t>> available_indices;
    std::mutex pool_mutex;

public:
    WhisperContextPool(size_t num_instances, const std::string& model_path, const whisper_context_params& params) {
        if (num_instances == 0) {
            throw std::invalid_argument("Pool must have at least one instance");
        }

        contexts.reserve(num_instances);
        available_indices.reset(new LockFreeQueue<size_t>());

        // Initialize contexts with CPU affinity
        for (size_t i = 0; i < num_instances; i++) {
            PaddedContext ctx;
            ctx.context = std::unique_ptr<WhisperContext>(new WhisperContext(model_path, params));
            ctx.core_affinity = i % std::thread::hardware_concurrency();
            contexts.push_back(std::move(ctx));
            available_indices->push(i);
        }
    }

    std::shared_ptr<Instance> get_instance() {
        size_t idx;
        // Try to get an available context index
        for (int attempt = 0; attempt < 5; attempt++) {
            if (available_indices->try_pop(idx)) {
                std::lock_guard<std::mutex> lock(pool_mutex);
                auto instance = std::make_shared<Instance>();
                instance->ctx = contexts[idx].context->get();
                in_use_indices.push_back(idx);
                return instance;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1 << attempt));
        }
        return nullptr;
    }

    void release_instance(std::shared_ptr<Instance>& instance) {
        if (!instance) return;

        std::lock_guard<std::mutex> lock(pool_mutex);
        // Find and remove the context index from in_use list
        for (auto it = in_use_indices.begin(); it != in_use_indices.end(); ++it) {
            if (contexts[*it].context->get() == instance->ctx) {
                available_indices->push(*it);
                in_use_indices.erase(it);
                instance.reset();
                return;
            }
        }
    }

    bool load_model(const std::string& model_path, const whisper_context_params& params) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        try {
            std::vector<PaddedContext> new_contexts;
            new_contexts.reserve(contexts.size());
            
            for (size_t i = 0; i < contexts.size(); i++) {
                PaddedContext ctx;
                ctx.context = std::unique_ptr<WhisperContext>(
                    new WhisperContext(model_path, params)
                );
                ctx.core_affinity = i % std::thread::hardware_concurrency();
                new_contexts.push_back(std::move(ctx));
            }

            contexts = std::move(new_contexts);
            
            // Reset available indices
            available_indices.reset(new LockFreeQueue<size_t>());
            for (size_t i = 0; i < contexts.size(); i++) {
                available_indices->push(i);
            }
            in_use_indices.clear();
            
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "Failed to load model: %s\n", e.what());
            return false;
        }
    }

    ~WhisperContextPool() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        contexts.clear();
        in_use_indices.clear();
        available_indices.reset();
    }
};