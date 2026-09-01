#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <random>
#include <atomic>
#include <memory>
#include <chrono>

template<typename T>
class LockFreeQueue {
    struct Node {
        std::shared_ptr<T> data;
        std::atomic<Node*> next;
        
        Node() : next(nullptr) {}
        explicit Node(const T& value) : data(std::make_shared<T>(value)), next(nullptr) {}
    };

    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    std::atomic<size_t> size_;

public:
    LockFreeQueue() : size_(0) {
        Node* dummy = new Node();
        head.store(dummy);
        tail.store(dummy);
    }

    ~LockFreeQueue() {
        Node* current = head.load();
        while (current) {
            Node* next = current->next.load();
            delete current;
            current = next;
        }
    }

    void push(const T& value) {
        Node* new_node = new Node(value);
        size_.fetch_add(1, std::memory_order_relaxed);
        
        while (true) {
            Node* last = tail.load(std::memory_order_acquire);
            Node* next = last->next.load(std::memory_order_acquire);
            
            if (last == tail.load(std::memory_order_acquire)) {
                if (next == nullptr) {
                    if (last->next.compare_exchange_weak(next, new_node,
                                                       std::memory_order_release,
                                                       std::memory_order_relaxed)) {
                        tail.compare_exchange_strong(last, new_node,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed);
                        return;
                    }
                } else {
                    tail.compare_exchange_strong(last, next,
                                               std::memory_order_release,
                                               std::memory_order_relaxed);
                }
            }
        }
    }

    bool try_pop(T& result) {
        while (true) {
            Node* first = head.load(std::memory_order_acquire);
            Node* last = tail.load(std::memory_order_acquire);
            Node* next = first->next.load(std::memory_order_acquire);
            
            if (first == head.load(std::memory_order_acquire)) {
                if (first == last) {
                    if (next == nullptr) {
                        return false;
                    }
                    tail.compare_exchange_strong(last, next,
                                               std::memory_order_release,
                                               std::memory_order_relaxed);
                } else {
                    if (next->data) {
                        result = *(next->data);
                        
                        if (head.compare_exchange_weak(first, next,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed)) {
                            size_.fetch_sub(1, std::memory_order_relaxed);
                            delete first;
                            return true;
                        }
                    }
                }
            }
        }
    }

    size_t size() const { return size_.load(std::memory_order_relaxed); }
    bool empty() const { return size() == 0; }
};

class WorkStealingThreadPool {
public:
    WorkStealingThreadPool(size_t num_threads) : queues(num_threads), stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i] {
                static constexpr size_t MAX_STEAL_ATTEMPTS = 3;
                static constexpr std::chrono::microseconds MIN_BACKOFF(1);
                static constexpr std::chrono::microseconds MAX_BACKOFF(100);

                while (true) {
                    std::function<void()> task;
                    bool found_task = false;

                    // First try to get task from own queue
                    if (queues[i].try_pop(task)) {
                        found_task = true;
                    }

                    // If no task in own queue, try to steal with exponential backoff
                    if (!found_task) {
                        auto backoff = MIN_BACKOFF;
                        
                        for (size_t attempt = 0; attempt < MAX_STEAL_ATTEMPTS; ++attempt) {
                            // Deterministic stealing pattern
                            size_t victim = (i + attempt + 1) % queues.size();
                            
                            if (queues[victim].try_pop(task)) {
                                found_task = true;
                                break;
                            }

                            // Exponential backoff between steal attempts
                            std::this_thread::sleep_for(backoff);
                            backoff = std::min(backoff * 2, MAX_BACKOFF);
                        }
                    }

                    // If still no task, wait for notification
                    if (!found_task) {
                        std::unique_lock<std::mutex> lock(wait_mutex);
                        condition.wait(lock, [this, &task, &found_task, i] {
                            if (stop && are_all_queues_empty()) {
                                return true;
                            }
                            if (queues[i].try_pop(task)) {
                                found_task = true;
                                return true;
                            }
                            return false;
                        });
                    }

                    if (stop && !found_task && are_all_queues_empty()) {
                        return;
                    }

                    if (found_task) {
                        task();
                        condition.notify_one();
                    }
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();

        // Use round-robin instead of random assignment
        static size_t next_queue = 0;
        size_t queue_idx;
        {
            std::unique_lock<std::mutex> lock(wait_mutex);
            queue_idx = next_queue;
            next_queue = (next_queue + 1) % queues.size();
        }

        if (stop) {
            throw std::runtime_error("enqueue on stopped WorkStealingThreadPool");
        }
        
        queues[queue_idx].push([task](){ (*task)(); });
        condition.notify_all();
        return res;
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(wait_mutex);
            stop = true;
        }
        
        condition.notify_all();
        
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    ~WorkStealingThreadPool() {
        if (!stop) {
            shutdown();
        }
    }

private:
    bool are_all_queues_empty() {
        for (auto& queue : queues) {
            if (!queue.empty()) {
                return false;
            }
        }
        return true;
    }

    std::vector<std::thread> workers;
    std::vector<LockFreeQueue<std::function<void()>>> queues;
    
    std::mutex wait_mutex;
    std::condition_variable condition;
    bool stop;
};