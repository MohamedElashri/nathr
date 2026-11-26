#include "allocation_counter.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>

namespace {
std::atomic<bool> counting = false;
std::atomic<std::size_t> allocations = 0;

void* allocate(std::size_t size) {
    if (counting.load(std::memory_order_relaxed))
        allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* ptr = std::malloc(std::max(size, std::size_t{1})))
        return ptr;
    throw std::bad_alloc();
}

void* allocate_aligned(std::size_t size, std::align_val_t alignment) {
    if (counting.load(std::memory_order_relaxed))
        allocations.fetch_add(1, std::memory_order_relaxed);
    void* ptr = nullptr;
    if (posix_memalign(&ptr, static_cast<std::size_t>(alignment), std::max(size, std::size_t{1})) == 0)
        return ptr;
    throw std::bad_alloc();
}
} // namespace

void* operator new(std::size_t size) {
    return allocate(size);
}
void* operator new[](std::size_t size) {
    return allocate(size);
}
void* operator new(std::size_t size, std::align_val_t a) {
    return allocate_aligned(size, a);
}
void* operator new[](std::size_t size, std::align_val_t a) {
    return allocate_aligned(size, a);
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}

void start_allocation_count() noexcept {
    allocations.store(0, std::memory_order_relaxed);
    counting.store(true, std::memory_order_release);
}
std::size_t stop_allocation_count() noexcept {
    counting.store(false, std::memory_order_release);
    return allocations.load(std::memory_order_relaxed);
}
