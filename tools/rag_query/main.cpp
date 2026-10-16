#include "rag/chunk.hpp"
#include "rag/context_builder.hpp"
#include "rag/embedder/nn.hpp"
#include "rag/turbovec.hpp"

#include <mln/runtime/runtime.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <vector>

namespace {
    struct IndexDeleter {
            void operator()(Tv::Index *index) const noexcept { tv_close(index); }
    };

    using Index = std::unique_ptr<Tv::Index, IndexDeleter>;

    std::vector<uint8_t> read_file(const std::filesystem::path &path) {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("could not open " + path.string());
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>());
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::println(stderr, "use: {} <embedder-artifact.mln> <asset-dir> <pergunta> [top_k]",
                     argv[0]);
        return 2;
    }

    try {
        const std::filesystem::path artifact(argv[1]);
        const std::filesystem::path assets(argv[2]);
        const std::string question = argv[3];
        const uint32_t top_k = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 5;

        Mln::Runtime runtime;
        auto session = runtime.session();
        session.load(artifact.string());

        const std::string payload = "query: " + question;
        Mln::Buffer input(payload.size());
        input.write(std::as_bytes(std::span(payload)));

        std::vector<float> embedding;
        session.invoke(input, [&](std::span<const uint8_t> output) {
            if (output.size() != Rag::Embedder::Arch::HIDDEN_SIZE * sizeof(float))
                throw std::runtime_error("invalid embedding size");
            embedding.resize(Rag::Embedder::Arch::HIDDEN_SIZE);
            std::memcpy(embedding.data(), output.data(), output.size());
        });

        double norm = 0.0;
        for (float value : embedding)
            norm += double(value) * value;
        std::println("norma L2 do vetor da consulta: {}", norm);

        if (const char *dump = std::getenv("RAG_DUMP_EMBEDDING"); dump) {
            for (float value : embedding)
                std::print("{} ", value);
            std::println();
            return 0;
        }
        std::println();

        const std::vector<uint8_t> image = read_file(assets / "rag_index.tvim");
        Index index{tv_open(image.data(), image.size())};
        if (!index)
            throw std::runtime_error(std::string("tv_open: ") + tv_last_error());

        std::vector<float> scores(top_k);
        std::vector<uint64_t> ids(top_k);
        uint32_t count = 0;
        if (tv_search(index.get(), embedding.data(), static_cast<uint32_t>(embedding.size()), top_k,
                      scores.data(), ids.data(), &count) != 0)
            throw std::runtime_error(std::string("tv_search: ") + tv_last_error());

        const std::vector<uint8_t> store = read_file(assets / "rag_chunks.bin");
        const Rag::Chunk::Store chunks = Rag::Chunk::Store::open(store);

        for (uint32_t i = 0; i < count; ++i) {
            Rag::Chunk chunk = chunks.at(ids[i]);
            std::println("[{}] score={}  id={}  fonte={} (+{})", i + 1, scores[i], ids[i],
                         chunk.source, chunk.offset);
            std::string preview = chunk.text.substr(0, 220);
            for (char &character : preview)
                if (character == '\n')
                    character = ' ';
            std::println("    {}...\n", preview);
        }

        const std::vector<Rag::Retrieved> retrieved =
            Rag::Context::select(question, chunks, ids, scores, 0.2F, true);

        std::println("contexto final:\n{}", Rag::Context::Builder{1000}.evidence(retrieved));
    } catch (const std::exception &error) {
        std::println(stderr, "error: {}", error.what());
        return 1;
    }

    return 0;
}
