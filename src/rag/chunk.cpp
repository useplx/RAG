#include "rag/chunk.hpp"

#include <algorithm>
#include <cstring>

namespace Rag {
    namespace {
        constexpr char MAGIC[8] = {'M', 'Y', 'R', 'A', 'G', 'C', 'H', 'K'};
        constexpr std::size_t HEADER = sizeof(MAGIC) + 2 * sizeof(uint32_t);
        constexpr std::size_t ENTRY = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t);

        template <typename T> T read_at(std::span<const uint8_t> bytes, std::size_t offset) {
            T value{};
            std::memcpy(&value, bytes.data() + offset, sizeof(T));
            return value;
        }

        template <typename T> void append(std::vector<uint8_t> &out, T value) {
            const auto *raw = reinterpret_cast<const uint8_t *>(&value);
            out.insert(out.end(), raw, raw + sizeof(T));
        }
    }

    Chunk::Store Chunk::Store::open(std::span<const uint8_t> bytes) {
        Store store;
        if (bytes.size() < HEADER || std::memcmp(bytes.data(), MAGIC, sizeof(MAGIC)) != 0)
            return store;
        if (read_at<uint32_t>(bytes, sizeof(MAGIC)) != VERSION)
            return store;

        const uint32_t count = read_at<uint32_t>(bytes, sizeof(MAGIC) + sizeof(uint32_t));
        const std::size_t table = HEADER;
        const std::size_t blob = table + std::size_t{count} * ENTRY;
        if (blob > bytes.size())
            return store;

        store.bytes_ = bytes;
        store.count_ = count;
        store.table_ = table;
        store.blob_ = blob;
        return store;
    }

    std::vector<uint8_t> Chunk::Store::build(std::vector<Chunk> chunks) {
        std::sort(chunks.begin(), chunks.end(),
                  [](const Chunk &left, const Chunk &right) { return left.id < right.id; });

        std::vector<uint8_t> out;
        out.insert(out.end(), MAGIC, MAGIC + sizeof(MAGIC));
        append<uint32_t>(out, VERSION);
        append<uint32_t>(out, static_cast<uint32_t>(chunks.size()));

        std::vector<uint8_t> blob;
        for (const Chunk &chunk : chunks) {
            append<uint64_t>(out, chunk.id);
            append<uint64_t>(out, chunk.offset);
            append<uint32_t>(out, static_cast<uint32_t>(chunk.text.size()));
            append<uint32_t>(out, static_cast<uint32_t>(chunk.source.size()));
            blob.insert(blob.end(), chunk.text.begin(), chunk.text.end());
            blob.insert(blob.end(), chunk.source.begin(), chunk.source.end());
        }

        out.insert(out.end(), blob.begin(), blob.end());
        return out;
    }

    Chunk Chunk::Store::at(uint64_t id) const {
        if (count_ == 0)
            return {};

        uint32_t low = 0;
        uint32_t high = count_;
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2;
            const uint64_t candidate =
                read_at<uint64_t>(bytes_, table_ + std::size_t{middle} * ENTRY);
            if (candidate == id) {
                std::size_t position = blob_;
                for (uint32_t i = 0; i < middle; ++i) {
                    const std::size_t entry = table_ + std::size_t{i} * ENTRY;
                    position += read_at<uint32_t>(bytes_, entry + 16);
                    position += read_at<uint32_t>(bytes_, entry + 20);
                }

                const std::size_t entry = table_ + std::size_t{middle} * ENTRY;
                const uint32_t text_len = read_at<uint32_t>(bytes_, entry + 16);
                const uint32_t source_len = read_at<uint32_t>(bytes_, entry + 20);
                if (position + text_len + source_len > bytes_.size())
                    return {};

                Chunk chunk;
                chunk.id = id;
                chunk.offset = read_at<uint64_t>(bytes_, entry + 8);
                chunk.text.assign(reinterpret_cast<const char *>(bytes_.data() + position),
                                  text_len);
                chunk.source.assign(
                    reinterpret_cast<const char *>(bytes_.data() + position + text_len),
                    source_len);
                return chunk;
            }
            if (candidate < id)
                low = middle + 1;
            else
                high = middle;
        }
        return {};
    }
}
