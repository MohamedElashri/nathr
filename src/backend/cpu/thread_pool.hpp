#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace nathr::detail {

inline constexpr unsigned max_cpu_threads = 64;

// The caller participates. Parallel submissions on a shared model are serialized;
// callbacks and work below the parallel threshold stay on their calling thread.
class ThreadPool {
  public:
    explicit ThreadPool(unsigned threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <class Operation>
    void rows(std::size_t count, std::size_t products_per_row, const Operation& operation) const {
        constexpr std::size_t minimum_products_per_thread = 32768;
        if (workers_.empty() || count < 16 ||
            count * products_per_row < minimum_products_per_thread * (workers_.size() + 1)) {
            operation(0, count);
            return;
        }
        run(count, &operation, [](const void* context, std::size_t begin, std::size_t end) {
            (*static_cast<const Operation*>(context))(begin, end);
        });
    }

  private:
    using Work = void (*)(const void*, std::size_t, std::size_t);
    void run(std::size_t count, const void* context, Work work) const;
    void worker(std::size_t lane);
    void invoke(std::size_t lane) const noexcept;
    void stop();

    mutable std::mutex submission_, mutex_;
    mutable std::condition_variable waiting_, completed_;
    mutable std::size_t generation_{}, pending_{}, count_{};
    mutable const void* context_{};
    mutable Work work_{};
    mutable std::exception_ptr error_;
    bool stopping_{};
    const std::size_t lanes_;
    std::vector<std::thread> workers_;
};

} // namespace nathr::detail
