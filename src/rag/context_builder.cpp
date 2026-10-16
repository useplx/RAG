#include "rag/context_builder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_set>
#include <utility>

namespace Rag {
    namespace {
        std::string normalized(std::string_view text) {
            std::string result;
            result.reserve(text.size());
            for (const unsigned char character : text)
                result.push_back(
                    std::isalnum(character) ? static_cast<char>(std::tolower(character)) : ' ');
            return result;
        }

        std::unordered_set<std::string> terms(std::string_view text) {
            static const std::unordered_set<std::string> stopwords = {
                "a",       "an",       "and",  "as",   "como",     "com",  "da",    "das",  "de",
                "do",      "does",     "dos",  "e",    "em",       "essa", "esse",  "esta", "este",
                "explain", "explique", "for",  "from", "funciona", "is",   "isso",  "it",   "no",
                "nos",     "o",        "of",   "os",   "para",     "que",  "sobre", "that", "the",
                "this",    "uma",      "what", "with", "works"};
            const std::string value = normalized(text);
            std::unordered_set<std::string> result;
            std::size_t begin = 0;
            while (begin < value.size()) {
                const std::size_t end = value.find(' ', begin);
                const std::string term =
                    value.substr(begin, (end == std::string::npos ? value.size() : end) - begin);
                if (term.size() >= 2 && !stopwords.contains(term))
                    result.insert(term);
                if (end == std::string::npos)
                    break;
                begin = end + 1;
            }
            return result;
        }

        std::size_t matches(const std::unordered_set<std::string> &query, std::string_view text) {
            const std::unordered_set<std::string> candidate = terms(text);
            return std::count_if(query.begin(), query.end(),
                                 [&](const std::string &term) { return candidate.contains(term); });
        }

        std::string_view utf8_prefix(std::string_view text, std::size_t bytes) {
            if (text.size() <= bytes)
                return text;
            while (bytes != 0 && (static_cast<unsigned char>(text[bytes]) & 0xc0U) == 0x80U)
                --bytes;
            return text.substr(0, bytes);
        }
    }

    float Context::score(std::string_view question, const Chunk &chunk, float similarity) {
        const std::unordered_set<std::string> query = terms(question);
        return similarity + 0.2F * static_cast<float>(matches(query, chunk.source)) +
               0.02F * static_cast<float>(matches(query, chunk.text));
    }

    std::vector<Retrieved> Context::select(std::string_view question, const Chunk::Store &store,
                                           std::span<const uint64_t> ids,
                                           std::span<const float> similarities,
                                           float minimum_similarity, bool require_lexical_match) {
        const std::size_t count = std::min(ids.size(), similarities.size());
        const std::unordered_set<std::string> query = terms(question);
        std::vector<Retrieved> selected;
        selected.reserve(count);
        std::unordered_set<uint64_t> seen_ids;
        std::unordered_set<std::string> seen_texts;

        for (std::size_t index = 0; index < count; ++index) {
            const float similarity = similarities[index];
            if (!std::isfinite(similarity) || similarity < minimum_similarity ||
                !seen_ids.insert(ids[index]).second)
                continue;

            Chunk chunk = store.at(ids[index]);
            if (chunk.id != ids[index] || chunk.text.empty() || chunk.source.empty() ||
                !seen_texts.insert(normalized(chunk.text)).second)
                continue;
            if (require_lexical_match && matches(query, chunk.source) == 0 &&
                matches(query, chunk.text) == 0)
                continue;

            selected.push_back(Retrieved{std::move(chunk), 0.0F});
            selected.back().score = score(question, selected.back().chunk, similarity);
        }

        std::sort(
            selected.begin(), selected.end(),
            [](const Retrieved &left, const Retrieved &right) { return left.score > right.score; });
        return selected;
    }

    std::string Context::Builder::evidence(const std::vector<Retrieved> &retrieved) const {
        if (retrieved.empty() || budget_ == 0)
            return {};

        std::vector<const Retrieved *> ordered;
        ordered.reserve(retrieved.size());
        for (const Retrieved &item : retrieved)
            if (!item.chunk.text.empty())
                ordered.push_back(&item);
        std::sort(ordered.begin(), ordered.end(),
                  [](const Retrieved *left, const Retrieved *right) {
                      return left->score > right->score;
                  });

        std::string message;
        std::size_t used = 0;
        int64_t number = 0;

        for (const Retrieved *item : ordered) {
            std::string prefix = "[" + std::to_string(number + 1) + "] ";
            if (!item->chunk.source.empty())
                prefix += item->chunk.source + "\n";
            constexpr std::string_view suffix = "\n\n";

            if (used + prefix.size() + suffix.size() >= budget_)
                continue;
            const std::size_t available = budget_ - used - prefix.size() - suffix.size();
            const std::string_view text = utf8_prefix(item->chunk.text, available);
            if (text.empty())
                continue;

            std::string entry = std::move(prefix);
            entry.append(text);
            entry.append(suffix);
            message += entry;
            used += entry.size();
            ++number;
        }

        return message;
    }

    std::string Context::Builder::chat_payload(const std::string &question,
                                               const std::vector<Retrieved> &retrieved) const {
        std::string payload = evidence(retrieved);
        if (payload.empty())
            return question;
        payload.push_back('\0');
        payload += question;
        return payload;
    }
}
