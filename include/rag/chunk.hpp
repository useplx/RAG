#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Rag {
    struct Chunk {
            class Store;
            uint64_t id = 0;
            std::string text;
            std::string source;
            uint64_t offset = 0;
    };

    class Chunk::Store {
        public:
            static constexpr uint32_t VERSION = 1;

            static Store open(std::span<const uint8_t> bytes);

            static std::vector<uint8_t> build(std::vector<Chunk> chunks);

            Chunk at(uint64_t id) const;

            uint32_t size() const { return count_; }

        private:
            std::span<const uint8_t> bytes_;
            uint32_t count_ = 0;
            std::size_t table_ = 0;
            std::size_t blob_ = 0;
    };
}
