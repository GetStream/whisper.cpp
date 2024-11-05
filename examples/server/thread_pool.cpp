#include "thread_pool.h"

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            const auto thread_id = std::this_thread::get_id();
            std::cout << "[" << get_current_time() << "] Worker " << thread_id
                      << " started\n";

            while(true) {
                std::function<void()> task;
                std::chrono::steady_clock::time_point enqueue_time;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });

                    if(this->stop && this->tasks.empty()) {
                        std::cout << "[" << get_current_time() << "] Worker " << thread_id
                                  << " shutting down\n";
                        return;
                    }

                    task = std::move(this->tasks.front().first);
                    enqueue_time = this->tasks.front().second;
                    this->tasks.pop();

                    std::cout << "[" << get_current_time() << "] Worker " << thread_id
                              << " dequeued task. Queue size now: " << this->tasks.size()
                              << ", Active workers: " << this->active_tasks.load() + 1 << "\n";
                }

                active_tasks++;
                auto start = std::chrono::steady_clock::now();
                auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    start - enqueue_time).count();

                total_wait_time.fetch_add(wait_time);

                std::cout << "[" << get_current_time() << "] Worker " << thread_id
                          << " starting task. Waited: " << wait_time << "ms\n";

                task();

                auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count();

                total_processing_time.fetch_add(process_time);
                total_tasks_processed++;
                active_tasks--;

                std::cout << "[" << get_current_time() << "] Worker " << thread_id
                          << " completed task. Processing time: " << process_time << "ms\n";
            }
        });
    }

    std::cout << "[" << get_current_time() << "] Thread pool initialized with "
              << threads << " workers\n";
}

void ThreadPool::shutdown() {
    std::cout << "[" << get_current_time() << "] Initiating thread pool shutdown\n";
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(auto &worker: workers) {
        if(worker.joinable()) {
            worker.join();
        }
    }
    std::cout << "[" << get_current_time() << "] Thread pool shutdown complete\n";
}

// Add the missing get_stats implementation
std::string ThreadPool::get_stats() const {
    std::ostringstream oss;
    size_t queued;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        queued = tasks.size();
    }

    oss << "ThreadPool Stats:\n"
        << "  Total Workers: " << get_total_workers() << "\n"
        << "  Active Workers: " << get_active_workers() << "\n"
        << "  Tasks Queued: " << queued << "\n"
        << "  Tasks Processed: " << get_total_tasks_processed() << "\n"
        << "  Avg Wait Time: " << get_average_wait_time() << "ms\n"
        << "  Avg Processing Time: " << get_average_processing_time() << "ms\n";

    return oss.str();
}
