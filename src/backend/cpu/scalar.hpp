#pragma once

#include "backend/cpu/kernels.hpp"
#include "core/input.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <variant>

namespace nathr::detail {

float half_to_float(std::uint16_t bits) noexcept;
std::uint16_t float_to_half(float value) noexcept;
inline constexpr float bf16_to_float(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(std::uint32_t{bits} << 16);
}

class CachePagePool {
  public:
    explicit CachePagePool(std::size_t byte_limit, std::function<bool()> reclaim = {})
        : limit_(byte_limit), reclaim_(std::move(reclaim)) {}

    template <class T> T* acquire(std::size_t elements) {
        const auto bytes = elements * sizeof(T);
        while (bytes > limit_ - in_use_) {
            if (!reclaim_ || !reclaim_())
                throw Error(ErrorCode::resource_limit, "KV page pool exhausted");
        }
        const auto found = std::ranges::find_if(free_, [&](const Page& page) {
            return page.bytes == bytes && std::holds_alternative<std::unique_ptr<T[]>>(page.data);
        });
        if (found != free_.end()) {
            auto* data = std::get<std::unique_ptr<T[]>>(found->data).release();
            free_.erase(found);
            ++reuses_;
            record_use(bytes);
            return data;
        }
        while (bytes > limit_ - resident_) {
            resident_ -= free_.front().bytes;
            free_.pop_front();
        }
        auto data = std::make_unique_for_overwrite<T[]>(elements);
        resident_ += bytes;
        record_use(bytes);
        return data.release();
    }

    template <class T> void release(T* pointer, std::size_t elements) noexcept {
        const auto bytes = elements * sizeof(T);
        in_use_ -= bytes;
        std::unique_ptr<T[]> page(pointer);
        try {
            free_.push_back({std::move(page), bytes});
        } catch (const std::bad_alloc&) {
            resident_ -= bytes;
        }
    }

    [[nodiscard]] std::size_t peak_bytes() const noexcept { return peak_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return in_use_; }
    [[nodiscard]] std::size_t resident_bytes() const noexcept { return resident_; }
    [[nodiscard]] std::size_t reuses() const noexcept { return reuses_; }

  private:
    struct Page {
        std::variant<std::unique_ptr<float[]>, std::unique_ptr<std::uint16_t[]>> data;
        std::size_t bytes;
    };

    void record_use(std::size_t bytes) noexcept {
        in_use_ += bytes;
        peak_ = std::max(peak_, in_use_);
    }

    std::list<Page> free_;
    const std::size_t limit_;
    const std::function<bool()> reclaim_;
    std::size_t in_use_ = 0, resident_ = 0, peak_ = 0, reuses_ = 0;
};

struct Cache {
    using Page = std::shared_ptr<void>;
    static constexpr std::size_t page_tokens = 16;
    [[nodiscard]] static std::size_t required_bytes(const ModelConfig& config, std::size_t capacity,
                                                    KvPrecision precision);
    [[nodiscard]] static std::size_t fit_tokens(const ModelConfig& config, std::size_t budget,
                                                KvPrecision precision);

    Cache(const ModelConfig& config, std::size_t capacity, KvPrecision precision = KvPrecision::f32,
          CachePagePool* pool = nullptr);
    // Allocate any missing pages before starting a forward pass.
    void ensure(std::size_t positions);
    [[nodiscard]] const Page& page(std::size_t index) const noexcept { return pages_[index]; }
    void use_prefix_page(std::size_t index, const Page& shared);

    template <class T>
    [[nodiscard]] std::span<const T> head_page(bool value, std::size_t layer, std::size_t head,
                                               std::size_t page) const {
        return page_rows<T>(value, layer, head, page);
    }

    template <class T>
    [[nodiscard]] std::span<T> row(bool value, std::size_t layer, std::size_t head, std::size_t position) {
        const auto page = position / page_tokens;
        assert(pages_[page].use_count() == 1);
        return page_rows<T>(value, layer, head, page).subspan(position % page_tokens * width, width);
    }

    [[nodiscard]] std::size_t bytes() const noexcept;
    const KvPrecision precision;
    const std::size_t capacity, kv_heads, width;

  private:
    template <class T>
    [[nodiscard]] std::span<T> page_rows(bool value, std::size_t layer, std::size_t head,
                                         std::size_t page) const {
        const auto positions = page_length(page);
        const auto offset =
            (value ? bank_width_ * positions : 0) + (layer * kv_heads + head) * positions * width;
        return {static_cast<T*>(pages_[page].get()) + offset, positions * width};
    }

    template <class T> struct Recycle {
        CachePagePool* pool;
        std::size_t elements;

        void operator()(T* data) const noexcept {
            if (pool)
                pool->release(data, elements);
            else
                delete[] data;
        }
    };
    [[nodiscard]] std::size_t page_length(std::size_t page) const noexcept;

    const std::size_t bank_width_, bytes_per_token_;
    CachePagePool* const pool_;
    std::vector<Page> pages_;
};

struct Scalar {
    static void embedding(TensorView table, std::size_t index, std::span<float> output);
    static void rms_norm(TensorView weight, std::span<const float> input, std::span<float> output,
                         float epsilon);
    static void layer_norm(TensorView gain, TensorView bias, std::span<const float> input,
                           std::span<float> output, float epsilon);
    static void rope(std::span<float> query, std::span<float> key, std::span<const float> frequency,
                     std::size_t position);
    static void store(Cache& cache, std::size_t layer, std::size_t position, std::span<const float> key,
                      std::span<const float> value);
    static void attention(const ModelConfig& config, std::size_t layer, std::size_t position, Cache* cache,
                          std::span<const float> keys, std::span<const float> values,
                          std::span<const float> query, std::span<float> output, std::span<float> scores);
    // Scores cover exactly the visible keys. Optional bias has one row per query head.
    static void attention_range(const ModelConfig& config, std::size_t layer, const Cache* cache,
                                std::span<const float> keys, std::span<const float> values,
                                std::span<const float> query, std::span<float> output,
                                std::span<float> scores, double scale, std::span<const float> bias = {});
    static void softmax(std::span<float> scores);
    static void swiglu(std::span<float> gate, std::span<const float> up);
    static void gelu(std::span<float> values);
    static void add_bias(std::span<float> output, TensorView bias);
    static void add(std::span<float> output, std::span<const float> residual);
};

} // namespace nathr::detail
