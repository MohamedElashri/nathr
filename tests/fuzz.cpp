#include <nathr/tokenizer.hpp>

#if defined(NATHR_FUZZ_gguf)
#include "formats/gguf.hpp"
#elif defined(NATHR_FUZZ_safetensors)
#include "formats/safetensors.hpp"
#elif defined(NATHR_FUZZ_quant)
#include "backend/cpu/kernels.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <unistd.h>

namespace {
using namespace nathr;

#if !defined(NATHR_FUZZ_quant)
class FuzzFile {
  public:
    FuzzFile() {
        auto pattern = (std::filesystem::temp_directory_path() / "nathr-fuzz-XXXXXX").string();
        const int descriptor = mkstemp(pattern.data());
        if (descriptor == -1)
            throw std::runtime_error("cannot create fuzz fixture");
        close(descriptor);
        path_ = pattern;
    }

    ~FuzzFile() { unlink(path_.c_str()); }

    const std::filesystem::path& replace(std::span<const std::uint8_t> data) const {
        std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        stream.close();
        return path_;
    }

  private:
    std::filesystem::path path_;
};
#endif

#if defined(NATHR_FUZZ_tokenizer)
void tokenizer_paths(std::span<const std::uint8_t> input) {
    std::string_view text(reinterpret_cast<const char*>(input.data()), input.size());
    Utf8Stream one, split;
    auto whole = one.push(text);
    whole += one.finish();
    auto parts = split.push(text.substr(0, text.size() / 2));
    parts += split.push(text.substr(text.size() / 2));
    parts += split.finish();
    if (whole != parts)
        std::abort();

    static const auto tokenizer =
        Tokenizer::load(std::filesystem::path(NATHR_FIXTURES) / "model/tokenizer.json");
    try {
        const auto ids = tokenizer.encode(text);
        if (std::ranges::any_of(ids, [&](Token id) { return id >= tokenizer.vocabulary(); }))
            std::abort();
        (void)tokenizer.decode(ids);
    } catch (const Error&) {
        // Arbitrary byte sequences need not be valid UTF-8.
    }
}
#endif
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const auto input = std::span(data, size);
#if defined(NATHR_FUZZ_quant)
    using namespace nathr::detail;
    if (input.empty())
        return 0;
    const auto type = static_cast<DType>(input[0] & 31);
    const auto format = encoding(type);
    if (!format || format->elements == 1 || size != 1 + format->bytes)
        return 0;
    // Give ASan an exact-size allocation and retain the unaligned leading byte.
    const std::vector<std::uint8_t> packed(input.begin(), input.end());
    const auto bytes = std::as_bytes(std::span(packed).subspan(1));
    auto scalar = scalar_kernel(type);
    std::vector<float> decoded(format->elements), activation(format->elements);
    for (std::size_t i = 0; i < activation.size(); ++i)
        activation[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 16.f;
    scalar.unpack(bytes.data(), decoded.data());
    const volatile float scalar_result = scalar.dot(bytes.data(), activation.data(), activation.size());
    (void)scalar_result;
    const volatile float selected_result = select_kernel(type, ExecutionPolicy::automatic, cpu_features())
                                               .dot(bytes.data(), activation.data(), activation.size());
    (void)selected_result;
#else
    static FuzzFile file;
    if (size > 65536)
        return 0;
#if defined(NATHR_FUZZ_tokenizer)
    if (size <= 2048)
        tokenizer_paths(input);
#endif
    try {
        const auto& path = file.replace(input);
#if defined(NATHR_FUZZ_safetensors)
        nathr::detail::SafeTensors parsed(path);
#elif defined(NATHR_FUZZ_gguf)
        nathr::detail::Gguf parsed(path);
        (void)parsed.config();
        (void)parsed.tokenizer_fingerprint();
#elif defined(NATHR_FUZZ_tokenizer)
        const auto parsed = Tokenizer::load(path);
        (void)parsed.decode(parsed.encode(" ab\n中文 <x>"));
#endif
    } catch (const Error&) {
        // Parser rejection is expected. Sanitizer findings propagate.
    }
#endif
    return 0;
}
