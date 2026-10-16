#include <mln/sdk/native.hpp>

#include "rag/embedder/nn.hpp"
#include "rag/errors.hpp"

#include <nlohmann/json.hpp>
#include <utf8proc.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {
    using namespace Rag::Embedder::Arch;

    constexpr const char *METASPACE = "\xe2\x96\x81";

    namespace Vocabulary {
        struct Data {
                std::unordered_map<std::string, std::pair<int32_t, float>> pieces;
                std::size_t longest = 0;
        };

        class Sax final : public nlohmann::json_sax<nlohmann::json> {
            public:
                Data vocabulary;
                std::string model_type;

                bool null() override { return primitive(); }
                bool boolean(bool) override { return primitive(); }
                bool number_integer(number_integer_t value) override {
                    return number(static_cast<float>(value));
                }
                bool number_unsigned(number_unsigned_t value) override {
                    return number(static_cast<float>(value));
                }
                bool number_float(number_float_t value, const string_t &) override {
                    return number(static_cast<float>(value));
                }
                bool string(string_t &value) override {
                    if (top() == Context::Model && key_ == "type")
                        model_type = value;
                    else if (top() == Context::VocabEntry && entry_index_ == 0)
                        piece_ = std::move(value);
                    return primitive();
                }
                bool binary(binary_t &) override { return primitive(); }

                bool start_object(std::size_t) override {
                    Context context = Context::Other;
                    if (stack_.empty())
                        context = Context::Root;
                    else if (top() == Context::Root && key_ == "model")
                        context = Context::Model;
                    stack_.push_back(context);
                    key_.clear();
                    return true;
                }
                bool key(string_t &value) override {
                    key_ = value;
                    return true;
                }
                bool end_object() override {
                    stack_.pop_back();
                    key_.clear();
                    return true;
                }
                bool start_array(std::size_t elements) override {
                    Context context = Context::Other;
                    if (top() == Context::Model && key_ == "vocab") {
                        context = Context::Vocab;
                        if (elements != static_cast<std::size_t>(-1))
                            vocabulary.pieces.reserve(elements);
                    } else if (top() == Context::Vocab) {
                        context = Context::VocabEntry;
                        entry_index_ = 0;
                        piece_.clear();
                        score_ = 0.0f;
                    }
                    stack_.push_back(context);
                    key_.clear();
                    return true;
                }
                bool end_array() override {
                    const Context context = top();
                    stack_.pop_back();
                    if (context == Context::VocabEntry) {
                        const int32_t id = next_id_++;
                        vocabulary.longest = std::max(vocabulary.longest, piece_.size());
                        vocabulary.pieces.emplace(std::move(piece_),
                                                  std::pair<int32_t, float>{id, score_});
                    }
                    key_.clear();
                    return true;
                }
                bool parse_error(std::size_t, const std::string &,
                                 const nlohmann::detail::exception &) override {
                    return false;
                }

            private:
                enum class Context { Root, Model, Vocab, VocabEntry, Other };
                std::vector<Context> stack_;
                std::string key_;
                std::string piece_;
                float score_ = 0.0f;
                std::size_t entry_index_ = 0;
                int32_t next_id_ = 0;

                Context top() const noexcept {
                    return stack_.empty() ? Context::Other : stack_.back();
                }
                bool primitive() {
                    if (top() == Context::VocabEntry)
                        ++entry_index_;
                    key_.clear();
                    return true;
                }
                bool number(float value) {
                    if (top() == Context::VocabEntry && entry_index_ == 1)
                        score_ = value;
                    return primitive();
                }
        };
    }

    std::mutex vocabulary_mutex;
    Vocabulary::Data cached;
    uint64_t cached_fingerprint = 0;
    const uint8_t *cached_data = nullptr;
    std::size_t cached_size = 0;

    uint64_t fingerprint(std::span<const uint8_t> bytes) {
        uint64_t value = 0xcbf29ce484222325ULL;
        for (uint8_t byte : bytes) {
            value ^= byte;
            value *= 0x100000001b3ULL;
        }
        return value ^ (bytes.size() * 0x9e3779b97f4a7c15ULL);
    }

    const Vocabulary::Data &vocabulary(std::span<const uint8_t> tokenizer_json) {
        if (!cached.pieces.empty() && tokenizer_json.data() == cached_data &&
            tokenizer_json.size() == cached_size)
            return cached;
        const uint64_t key = fingerprint(tokenizer_json);
        if (key == cached_fingerprint) {
            cached_data = tokenizer_json.data();
            cached_size = tokenizer_json.size();
            return cached;
        }

        Vocabulary::Sax sax;
        if (!nlohmann::json::sax_parse(tokenizer_json.begin(), tokenizer_json.end(), &sax))
            Mln::raise(Rag::Errors::Embedder::Tokenizer::Json::Invalid);
        if (sax.model_type != "Unigram")
            Mln::raise(Rag::Errors::Embedder::Tokenizer::Model::Unsupported, sax.model_type);
        if (sax.vocabulary.pieces.empty())
            Mln::raise(Rag::Errors::Embedder::Tokenizer::Vocabulary::Empty);

        cached = std::move(sax.vocabulary);
        cached_fingerprint = key;
        cached_data = tokenizer_json.data();
        cached_size = tokenizer_json.size();
        return cached;
    }

    std::string normalize(std::string_view text) {
        utf8proc_uint8_t *output = nullptr;
        const utf8proc_ssize_t size =
            utf8proc_map(reinterpret_cast<const utf8proc_uint8_t *>(text.data()),
                         static_cast<utf8proc_ssize_t>(text.size()), &output,
                         static_cast<utf8proc_option_t>(UTF8PROC_COMPOSE | UTF8PROC_COMPAT));
        if (size < 0 || !output)
            return std::string(text);
        const std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> owned(output, &std::free);
        std::string normalized(reinterpret_cast<const char *>(owned.get()),
                               static_cast<std::size_t>(size));
        return normalized;
    }

    std::string metaspace(const std::string &text) {
        std::string out = METASPACE;
        for (char character : text) {
            if (character == ' ')
                out += METASPACE;
            else
                out.push_back(character);
        }
        return out;
    }

    std::vector<int32_t> segment(const Vocabulary::Data &vocabulary, const std::string &text) {
        constexpr float NEGATIVE = -std::numeric_limits<float>::infinity();
        const std::size_t length = text.size();

        std::vector<float> best(length + 1, NEGATIVE);
        std::vector<std::size_t> previous(length + 1, 0);
        std::vector<int32_t> token(length + 1, UNK_TOKEN_ID);
        best[0] = 0.0f;

        for (std::size_t start = 0; start < length; ++start) {
            if (best[start] == NEGATIVE)
                continue;

            const std::size_t limit = std::min(length, start + vocabulary.longest);
            bool matched = false;

            for (std::size_t end = start + 1; end <= limit; ++end) {
                if (end < length && (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80)
                    continue;

                const auto found = vocabulary.pieces.find(text.substr(start, end - start));
                if (found == vocabulary.pieces.end())
                    continue;

                matched = true;
                const float score = best[start] + found->second.second;
                if (score > best[end]) {
                    best[end] = score;
                    previous[end] = start;
                    token[end] = found->second.first;
                }
            }

            if (matched)
                continue;

            std::size_t end = start + 1;
            while (end < length && (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80)
                ++end;
            const float score = best[start] - 10.0f;
            if (score > best[end]) {
                best[end] = score;
                previous[end] = start;
                token[end] = UNK_TOKEN_ID;
            }
        }

        std::vector<int32_t> ids;
        for (std::size_t position = length; position > 0; position = previous[position])
            ids.push_back(token[position]);
        std::reverse(ids.begin(), ids.end());
        return ids;
    }
}

PUBLISH(rag, embed_tokenize, std::vector<uint8_t>, std::span<const uint8_t> text,
        std::span<const uint8_t> tokenizer_json) {
    std::lock_guard lock(vocabulary_mutex);
    const Vocabulary::Data &pieces = vocabulary(tokenizer_json);

    const std::string prepared = metaspace(
        normalize(std::string_view(reinterpret_cast<const char *>(text.data()), text.size())));

    std::vector<int32_t> ids = segment(pieces, prepared);

    const std::size_t room = static_cast<std::size_t>(SEQUENCE) - 2;
    if (ids.size() > room)
        ids.resize(room);
    ids.insert(ids.begin(), static_cast<int32_t>(BOS_TOKEN_ID));
    ids.push_back(static_cast<int32_t>(EOS_TOKEN_ID));

    std::vector<float> tokens(SEQUENCE, static_cast<float>(PAD_TOKEN_ID));

    std::vector<uint16_t> mask(SEQUENCE, 0x0000);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        tokens[i] = static_cast<float>(ids[i]);
        mask[i] = 0x3c00;
    }

    std::vector<uint8_t> out(SEQUENCE * sizeof(float) + SEQUENCE * sizeof(uint16_t));
    std::memcpy(out.data(), tokens.data(), SEQUENCE * sizeof(float));
    std::memcpy(out.data() + SEQUENCE * sizeof(float), mask.data(), SEQUENCE * sizeof(uint16_t));
    return out;
}
