#include "rag/chunk.hpp"
#include "rag/embedder/nn.hpp"
#include "rag/turbovec.hpp"

#include <mln/runtime/runtime.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {
    constexpr std::size_t CHUNK_CHARS = 700;
    constexpr std::size_t CHUNK_OVERLAP = 120;

    struct IndexDeleter {
            void operator()(Tv::Index *index) const noexcept {
                if (index != nullptr)
                    tv_close(index);
            }
    };

    using Index = std::unique_ptr<Tv::Index, IndexDeleter>;

    std::string read_file(const std::filesystem::path &path) {
        std::ifstream file(path, std::ios::binary);
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    std::filesystem::path common_parent(const std::vector<std::filesystem::path> &paths) {
        std::filesystem::path root = paths.front().parent_path();
        for (const std::filesystem::path &path : paths) {
            const std::filesystem::path parent = path.parent_path();
            std::filesystem::path common;
            auto left = root.begin();
            auto right = parent.begin();
            while (left != root.end() && right != parent.end() && *left == *right) {
                common /= *left;
                ++left;
                ++right;
            }
            root = std::move(common);
        }
        return root;
    }

    uint64_t identifier(const std::string &source, std::size_t offset) {
        uint64_t value = 0xcbf29ce484222325ULL;
        for (char character : source) {
            value ^= static_cast<unsigned char>(character);
            value *= 0x100000001b3ULL;
        }
        value ^= offset + 0x9e3779b97f4a7c15ULL;
        value *= 0x100000001b3ULL;
        return value == 0 ? 1 : value;
    }

    std::vector<Rag::Chunk> split(const std::string &text, const std::string &source) {
        std::vector<Rag::Chunk> chunks;
        std::size_t start = 0;

        while (start < text.size()) {
            std::size_t length = std::min(CHUNK_CHARS, text.size() - start);
            if (start + length < text.size()) {
                const std::size_t paragraph = text.rfind("\n\n", start + length);
                if (paragraph != std::string::npos && paragraph > start + 200)
                    length = paragraph - start;
            }

            std::string body = text.substr(start, length);
            if (body.find_first_not_of(" \t\r\n") != std::string::npos)
                chunks.push_back(Rag::Chunk{identifier(source, start), std::move(body), source,
                                            static_cast<uint64_t>(start)});

            if (start + length >= text.size())
                break;
            start += length > CHUNK_OVERLAP ? length - CHUNK_OVERLAP : length;
        }
        return chunks;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::println(stderr, "usage: {} <artifact.mln> <output-dir> <document>...", argv[0]);
        return 2;
    }

    try {
        const std::filesystem::path artifact(argv[1]);
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);

        if (!std::filesystem::exists(artifact))
            throw std::runtime_error("artifact does not exist: " + artifact.string());

        Mln::Runtime runtime;
        auto session = runtime.session();
        session.load(artifact.string());

        std::vector<std::filesystem::path> documents;
        documents.reserve(static_cast<std::size_t>(argc - 3));
        for (int i = 3; i < argc; ++i) {
            documents.push_back(std::filesystem::weakly_canonical(argv[i]));
        }

        const std::filesystem::path root = common_parent(documents);
        std::vector<Rag::Chunk> chunks;
        for (const std::filesystem::path &path : documents) {
            const std::string source = path.lexically_relative(root).generic_string();
            std::vector<Rag::Chunk> parts = split(read_file(path), source);
            std::println("{}: {} chunks", source, parts.size());
            chunks.insert(chunks.end(), std::make_move_iterator(parts.begin()),
                          std::make_move_iterator(parts.end()));
        }
        if (chunks.empty())
            throw std::runtime_error("no chunks were produced");

        Index index(tv_create(static_cast<uint32_t>(Rag::Embedder::Arch::HIDDEN_SIZE), 4));
        if (!index)
            throw std::runtime_error(std::string("tv_create: ") + tv_last_error());

        std::vector<float> vectors;
        std::vector<uint64_t> ids;
        vectors.reserve(chunks.size() * Rag::Embedder::Arch::HIDDEN_SIZE);

        for (std::size_t i = 0; i < chunks.size(); ++i) {
            const std::string payload = "passage: " + chunks[i].text;
            Mln::Buffer input(payload.size());
            input.write(std::as_bytes(std::span(payload)));

            const std::size_t base = vectors.size();
            session.invoke(input, [&](std::span<const uint8_t> output) {
                const std::size_t expected = Rag::Embedder::Arch::HIDDEN_SIZE * sizeof(float);
                if (output.size() != expected)
                    throw std::runtime_error("o encoder devolveu " + std::to_string(output.size()) +
                                             " bytes, expected " + std::to_string(expected));
                vectors.resize(base + Rag::Embedder::Arch::HIDDEN_SIZE);
                std::memcpy(vectors.data() + base, output.data(), output.size());
            });

            ids.push_back(chunks[i].id);
            if ((i + 1) % 20 == 0 || i + 1 == chunks.size()) {
                std::print("  embutidos {}/{}\r", i + 1, chunks.size());
                std::fflush(stdout);
            }
        }
        std::println();

        if (tv_add(index.get(), vectors.data(), static_cast<uint32_t>(ids.size()),
                   static_cast<uint32_t>(Rag::Embedder::Arch::HIDDEN_SIZE), ids.data()) != 0)
            throw std::runtime_error(std::string("tv_add: ") + tv_last_error());

        const std::size_t needed = tv_to_bytes(index.get(), nullptr, 0);
        std::vector<uint8_t> image(needed);
        tv_to_bytes(index.get(), image.data(), image.size());

        const std::filesystem::path index_path = output / "rag_index.tvim";
        std::ofstream(index_path, std::ios::binary)
            .write(reinterpret_cast<const char *>(image.data()),
                   static_cast<std::streamsize>(image.size()));

        const std::vector<uint8_t> store = Rag::Chunk::Store::build(chunks);
        const std::filesystem::path chunks_path = output / "rag_chunks.bin";
        std::ofstream(chunks_path, std::ios::binary)
            .write(reinterpret_cast<const char *>(store.data()),
                   static_cast<std::streamsize>(store.size()));

        std::println("index:   {} ({} bytes, {} vectors)\nchunks:  {} ({} bytes)",
                     index_path.string(), image.size(), ids.size(), chunks_path.string(),
                     store.size());
    } catch (const std::exception &error) {
        std::println(stderr, "error: {}", error.what());
        return 1;
    }

    return 0;
}
