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
                auto instance = std::make_shared<Instance>();
                instance->ctx = contexts[idx].context->get();
                in_use_indices.push_back(idx);
                return instance;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1 << attempt));
        }
        throw std::runtime_error("No available whisper contexts");
    }

    void release_instance(std::shared_ptr<Instance>& instance) {
        if (!instance) return;

        // Find and remove the context index from in_use list
        for (auto it = in_use_indices.begin(); it != in_use_indices.end(); ++it) {
            if (contexts[*it].context->get() == instance->ctx) {
                available_indices->push(*it);
                in_use_indices.erase(it);
                instance.reset();
                return;  // Add return here to exit after finding the match
            }
        }
    }

    void reset() {
        contexts.clear();
        in_use_indices.clear();
        available_indices.reset();
    }

private:
    std::vector<PaddedContext> contexts;
    std::vector<size_t> in_use_indices;
    std::unique_ptr<LockFreeQueue<size_t>> available_indices;
};