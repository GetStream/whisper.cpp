#include "thread_pool.h"

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            const auto thread_id = std::this_thread::get_id();
            std::cout << "[" << get_current_time() << "] Worker " << thread_id
                      << " started\n";
            while(true) {
                std::shared_ptr<std::packaged_task<void()>> task;
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

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                (*task)();
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
