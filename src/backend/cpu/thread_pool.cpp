#include "backend/cpu/thread_pool.hpp"

#include <nathr/nathr.hpp>

#include <system_error>

namespace nathr::detail {

ThreadPool::ThreadPool(unsigned threads) : lanes_(threads) {
    if (!threads || threads > max_cpu_threads)
        throw Error(ErrorCode::invalid_argument, "CPU thread count must be between 1 and 64");
    workers_.reserve(threads - 1);
    try {
        for (std::size_t lane = 1; lane < lanes_; ++lane)
            workers_.emplace_back([this, lane] { worker(lane); });
    } catch (const std::system_error& error) {
        stop();
        throw Error(ErrorCode::backend, "cannot create CPU workers: " + std::string(error.what()));
    }
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::stop() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    waiting_.notify_all();
    for (auto& worker : workers_)
        worker.join();
}

void ThreadPool::invoke(std::size_t lane) const noexcept {
    const auto begin = lane * count_ / lanes_;
    const auto end = (lane + 1) * count_ / lanes_;
    try {
        work_(context_, begin, end);
    } catch (...) {
        std::lock_guard lock(mutex_);
        if (!error_)
            error_ = std::current_exception();
    }
}

void ThreadPool::worker(std::size_t lane) {
    std::size_t seen = 0;
    std::unique_lock lock(mutex_);
    while (true) {
        waiting_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_)
            return;
        seen = generation_;
        lock.unlock();
        invoke(lane);
        lock.lock();
        if (--pending_ == 0)
            completed_.notify_one();
    }
}

void ThreadPool::run(std::size_t count, const void* context, Work work) const {
    std::lock_guard submit(submission_);
    std::unique_lock lock(mutex_);
    count_ = count;
    context_ = context;
    work_ = work;
    error_ = nullptr;
    pending_ = workers_.size();
    ++generation_;
    lock.unlock();
    waiting_.notify_all();
    invoke(0);
    lock.lock();
    completed_.wait(lock, [&] { return pending_ == 0; });
    if (error_)
        std::rethrow_exception(error_);
}

} // namespace nathr::detail
