#include "rag/embedder/orchestration.hpp"
#include "rag/errors.hpp"

#include <mln/sdk/native.hpp>

#include <array>
#include <vector>

using namespace Mln::Sdk;
using namespace Rag::Embedder::Arch;

namespace Rag::Embedder {
    IMPORT(rag, embed_tokenize, Mln::Buffer(Mln::Buffer, Mln::Buffer));

    Mln::Buffer Orchestration::forward(Mln::Runtime::Session &session, Mln::Buffer input) {
        Mln::Buffer tokenizer = session.asset("embedder_tokenizer.json");
        Mln::Buffer encoded = embed_tokenize(std::move(input), std::move(tokenizer));

        constexpr uint32_t ids_bytes = static_cast<uint32_t>(SEQUENCE) * sizeof(float);
        constexpr uint32_t mask_bytes = static_cast<uint32_t>(SEQUENCE) * sizeof(uint16_t);

        std::vector<uint8_t> values(ids_bytes + mask_bytes);
        encoded.read(std::as_writable_bytes(std::span(values)));

        Mln::Buffer tokens(ids_bytes);
        tokens.write(std::as_bytes(std::span(values.data(), ids_bytes)));
        Mln::Buffer mask(mask_bytes);
        mask.write(std::as_bytes(std::span(values.data() + ids_bytes, mask_bytes)));

        std::array<Mln::Buffer, 2> inputs{std::move(tokens), std::move(mask)};
        std::vector<Mln::Buffer> output = session.invoke("embedder.forward", inputs);
        if (output.size() != 1)
            Mln::raise(Errors::Embedder::Runtime::Output::Count::Invalid, output.size());
        return std::move(output.front());
    }
}
