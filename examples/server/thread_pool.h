#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <atomic>
#include <sstream>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <utility>

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool() { shutdown(); }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        const auto enqueue_time = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if(stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            std::cout << "[" << get_current_time() << "] Enqueueing task. Current queue size: "
                      << tasks.size() << ", Active workers: " << active_tasks.load() << "\n";

            tasks.emplace(std::make_pair(
                [task](){ (*task)(); },
                enqueue_time
            ));
        }

        condition.notify_one();
        return res;
    }

    void shutdown();

    size_t get_tasks_queued() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return tasks.size();
    }

    size_t get_active_workers() const {
        return active_tasks.load();
    }

    size_t get_total_workers() const {
        return workers.size();
    }

    uint64_t get_total_tasks_processed() const {
        return total_tasks_processed.load();
    }

    double get_average_wait_time() const {
        uint64_t processed = total_tasks_processed.load();
        if(processed == 0) return 0.0;
        return static_cast<double>(total_wait_time.load()) / processed;
    }

    double get_average_processing_time() const {
        uint64_t processed = total_tasks_processed.load();
        if(processed == 0) return 0.0;
        return static_cast<double>(total_processing_time.load()) / processed;
    }

    std::string get_stats() const;

private:
    std::string get_current_time() const {
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
        return ss.str();
    }

    std::vector<std::thread> workers;
    std::queue<std::pair<std::function<void()>, std::chrono::steady_clock::time_point>> tasks;

    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    std::atomic<size_t> active_tasks{0};
    std::atomic<uint64_t> total_tasks_processed{0};
    std::atomic<uint64_t> total_wait_time{0};
    std::atomic<uint64_t> total_processing_time{0};
};

#endif // THREAD_POOL_H
