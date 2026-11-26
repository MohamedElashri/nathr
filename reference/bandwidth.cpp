// One-thread copy bandwidth probe. The reported traffic is logical read + write bytes.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>

int main() {
    constexpr std::size_t bytes = 256 * 1024 * 1024;
    constexpr int warmups = 2;
    constexpr int repetitions = 7;
    auto source = std::make_unique<unsigned char[]>(bytes);
    auto target = std::make_unique<unsigned char[]>(bytes);
    std::fill_n(source.get(), bytes, 49);
    std::fill_n(target.get(), bytes, 98);
    std::cout << "{\"bytes_per_buffer\":" << bytes << ",\"warmups\":" << warmups << ",\"logical_gb_s\":[";
    for (int i = -warmups; i < repetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        std::memcpy(target.get(), source.get(), bytes);
        // Force the copy to complete before timing and prevent dead-store elimination.
        asm volatile("" : : "r"(target.get()) : "memory");
        const auto end = std::chrono::steady_clock::now();
        if (i >= 0) {
            if (i != 0) {
                std::cout << ',';
            }
            const double seconds = std::chrono::duration<double>(end - start).count();
            std::cout << 2.0 * bytes / seconds / 1e9;
        }
        if (target[bytes / 2] != 49) {
            return 1;
        }
    }
    std::cout << "]}\n";
}
