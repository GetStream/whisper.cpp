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
#include <type_traits>
#include <memory>

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool() { shutdown(); }

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
        using return_type = typename std::result_of<F()>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::forward<F>(f)
        );

        std::future<return_type> res = task->get_future();

        auto wrapper = std::make_shared<std::packaged_task<void()>>(
            [task]() { (*task)(); }
        );

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.push(wrapper);
        }
        condition.notify_one();
        return res;
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        using return_type = typename std::result_of<decltype(bound)()>::type;

        return enqueue([bound]() { return bound(); });
    }

    void shutdown();

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
    std::queue<std::shared_ptr<std::packaged_task<void()>>> tasks;
    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

#endif // THREAD_POOL_H
