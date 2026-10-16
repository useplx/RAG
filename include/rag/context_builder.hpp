#pragma once

#include "rag/chunk.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Rag {
    struct Retrieved {
            Chunk chunk;
            float score = 0.0f;
    };

    namespace Context {
        float score(std::string_view question, const Chunk &chunk, float similarity);

        std::vector<Retrieved> select(std::string_view question, const Chunk::Store &store,
                                      std::span<const uint64_t> ids,
                                      std::span<const float> similarities, float minimum_similarity,
                                      bool require_lexical_match);

        class Builder {
            public:
                explicit Builder(std::size_t budget = 8000) : budget_(budget) {}

                std::string evidence(const std::vector<Retrieved> &retrieved) const;

                std::string chat_payload(const std::string &question,
                                         const std::vector<Retrieved> &retrieved) const;

            private:
                std::size_t budget_;
        };
    }
}
