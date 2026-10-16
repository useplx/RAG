#include "rag/chunk.hpp"
#include "rag/context_builder.hpp"

#include <cassert>
#include <cmath>
#include <span>
#include <string>
#include <vector>

int main() {
    const std::vector<uint8_t> bytes = Rag::Chunk::Store::build({
        Rag::Chunk{1, "MyeLiN creates self-contained artifacts.", "myelin.md", 0},
        Rag::Chunk{2, "Convolution is not covered by this document.", "unrelated.md", 0},
    });
    const Rag::Chunk::Store store = Rag::Chunk::Store::open(bytes);

    const std::vector<uint64_t> ids{1, 1, 2, 99};
    const std::vector<float> scores{0.84F, 0.83F, 0.90F, 0.99F};
    const std::vector<Rag::Retrieved> selected =
        Rag::Context::select("What is MyeLiN?", store, ids, scores, 0.75F, true);
    assert(selected.size() == 1);
    assert(selected.front().chunk.id == 1);

    const std::vector<float> invalid_score{std::nanf("")};
    assert(Rag::Context::select("MyeLiN", store, std::span(ids).first(1), invalid_score, 0.0F, true)
               .empty());
    assert(Rag::Context::select("CNN", store, ids, scores, 0.75F, true).empty());

    const Rag::Context::Builder builder{64};
    const std::string payload = builder.chat_payload("What is MyeLiN?", selected);
    assert(payload.find('\0') != std::string::npos);
    assert(payload.size() <= 64 + std::string("What is MyeLiN?").size() + 1);

    const std::string fallback = builder.chat_payload("What is a CNN?", {});
    assert(fallback == "What is a CNN?");
    assert(fallback.find('\0') == std::string::npos);
}
