#include "rag/runtime_hook.hpp"
#include "rag/chunk.hpp"
#include "rag/context_builder.hpp"
#include "rag/embedder/nn.hpp"
#include "rag/errors.hpp"

#include <mln/sdk/native.hpp>

#include <array>
#include <cstring>
#include <string>
#include <vector>

#ifndef RAG_CONTEXT_BUDGET
#define RAG_CONTEXT_BUDGET 2200
#endif

#ifndef RAG_EMBEDDER_ENTRYPOINT
#define RAG_EMBEDDER_ENTRYPOINT "embedder.forward"
#endif

#ifndef RAG_TOP_K
#define RAG_TOP_K 3
#endif

#ifndef RAG_MINIMUM_SIMILARITY
#define RAG_MINIMUM_SIMILARITY 0.2F
#endif

#ifndef RAG_REQUIRE_LEXICAL_MATCH
#define RAG_REQUIRE_LEXICAL_MATCH true
#endif

using namespace Mln::Sdk;

namespace Rag::Hook {
    IMPORT(rag, embed_tokenize, Mln::Buffer(Mln::Buffer, Mln::Buffer));
    IMPORT(rag, index_open, uint64_t(Mln::Buffer));
    IMPORT(rag, index_search, Mln::Buffer(uint64_t, uint32_t, Mln::Buffer));

    namespace {
        std::string read_text(const Mln::Buffer &buffer) {
            std::string text(buffer.size(), '\0');
            if (!buffer.empty())
                buffer.read(std::as_writable_bytes(std::span(text)));
            return text;
        }

        bool is_text(std::string_view value) { return value.find('\0') == std::string_view::npos; }

        Mln::Buffer write_text(const std::string &text) {
            Mln::Buffer buffer(text.size());
            if (!text.empty())
                buffer.write(std::as_bytes(std::span(text)));
            return buffer;
        }

    }

    Mln::Buffer augment(Mln::Runtime::Session &session, Mln::Buffer input) {
        const std::string question = read_text(input);
        if (!is_text(question))
            return input;

        Mln::Buffer tokenizer = session.asset("embedder_tokenizer.json");
        Mln::Buffer query = write_text("query: " + question);
        Mln::Buffer encoded = embed_tokenize(std::move(query), std::move(tokenizer));

        constexpr uint32_t window = static_cast<uint32_t>(Embedder::Arch::SEQUENCE);
        constexpr uint32_t ids_bytes = window * sizeof(float);
        constexpr uint32_t mask_bytes = window * sizeof(uint16_t);

        std::vector<uint8_t> values(ids_bytes + mask_bytes);
        encoded.read(std::as_writable_bytes(std::span(values)));

        Mln::Buffer tokens(ids_bytes);
        tokens.write(std::as_bytes(std::span(values.data(), ids_bytes)));
        Mln::Buffer mask(mask_bytes);
        mask.write(std::as_bytes(std::span(values.data() + ids_bytes, mask_bytes)));

        std::array<Mln::Buffer, 2> inputs{std::move(tokens), std::move(mask)};
        std::vector<Mln::Buffer> embedding = session.invoke(RAG_EMBEDDER_ENTRYPOINT, inputs);
        if (embedding.size() != 1)
            Mln::raise(Errors::Hook::Embedding::Output::Count::Invalid, embedding.size());

        Mln::Buffer index_asset = session.asset("rag_index.tvim");
        const uint64_t index = index_open(std::move(index_asset));

        Mln::Buffer embedding_buffer = std::move(embedding.front());
        Mln::Buffer results =
            index_search(index, static_cast<uint32_t>(RAG_TOP_K), std::move(embedding_buffer));

        uint32_t count = 0;
        results.read(std::as_writable_bytes(std::span(&count, 1)));
        if (count > RAG_TOP_K)
            Mln::raise(Errors::Hook::Search::Result::Count::Invalid, count, RAG_TOP_K);
        const std::size_t expected = sizeof(count) + count * (sizeof(uint64_t) + sizeof(float));
        if (results.size() != expected)
            Mln::raise(Errors::Hook::Search::Result::Storage::Invalid, results.size(), expected,
                       count);
        std::vector<uint64_t> ids(count);
        std::vector<float> scores(count);
        if (count != 0) {
            results.read(sizeof(count), std::as_writable_bytes(std::span(ids)));
            results.read(sizeof(count) + count * sizeof(uint64_t),
                         std::as_writable_bytes(std::span(scores)));
        }

        Mln::Buffer chunks_asset = session.asset("rag_chunks.bin");
        std::vector<uint8_t> chunk_bytes(chunks_asset.size());
        if (!chunk_bytes.empty())
            chunks_asset.read(std::as_writable_bytes(std::span(chunk_bytes)));

        const Chunk::Store store = Chunk::Store::open(chunk_bytes);
        std::vector<Retrieved> passages = Context::select(
            question, store, ids, scores, RAG_MINIMUM_SIMILARITY, RAG_REQUIRE_LEXICAL_MATCH);

        return write_text(Context::Builder{RAG_CONTEXT_BUDGET}.chat_payload(question, passages));
    }
}
