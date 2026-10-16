#include <mln/sdk/native.hpp>

#include "rag/errors.hpp"
#include "rag/turbovec.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
    namespace Index {
        struct Deleter {
                void operator()(Tv::Index *index) const noexcept {
                    if (index)
                        tv_close(index);
                }
        };

        using Pointer = std::unique_ptr<Tv::Index, Deleter>;
    }

    std::mutex registry_mutex;
    std::unordered_map<uint64_t, Index::Pointer> indices;
    std::unordered_map<uint64_t, uint64_t> opened;
    std::unordered_set<uint64_t> persistent;
    uint64_t next_handle = 1;

    uint64_t fingerprint(std::span<const uint8_t> bytes) {
        uint64_t value = 0xcbf29ce484222325ULL;
        for (uint8_t byte : bytes) {
            value ^= byte;
            value *= 0x100000001b3ULL;
        }
        return value ^ (bytes.size() * 0x9e3779b97f4a7c15ULL);
    }

    std::string error_detail() {
        const char *detail = tv_last_error();
        return detail && *detail ? std::string(detail) : std::string("sem detalhe do provider");
    }

    Tv::Index *handle_of(uint64_t handle) {
        if (handle == 0)
            Mln::raise(Rag::Errors::Index::Handle::Null);
        const auto found = indices.find(handle);
        if (found == indices.end())
            Mln::raise(Rag::Errors::Index::Handle::Unknown, handle);
        return found->second.get();
    }

    uint64_t store(Index::Pointer index) {
        while (next_handle == 0 || indices.contains(next_handle))
            ++next_handle;
        const uint64_t handle = next_handle++;
        indices.emplace(handle, std::move(index));
        return handle;
    }
}

PUBLISH(rag, index_open, uint64_t, std::span<const uint8_t> index_bytes) {
    if (index_bytes.empty())
        Mln::raise(Rag::Errors::Index::Asset::Empty);

    const uint64_t key = fingerprint(index_bytes);
    std::lock_guard lock(registry_mutex);

    if (const auto found = opened.find(key); found != opened.end())
        return found->second;

    Index::Pointer index(tv_open(index_bytes.data(), index_bytes.size()));
    if (!index)
        Mln::raise(Rag::Errors::Index::Asset::Load::Failed, error_detail());
    const uint64_t handle = store(std::move(index));
    opened.emplace(key, handle);
    persistent.insert(handle);
    return handle;
}

PUBLISH(rag, index_create, uint64_t, uint32_t dim, uint32_t bit_width) {
    Index::Pointer index(tv_create(dim, bit_width));
    if (!index)
        Mln::raise(Rag::Errors::Index::Create::Failed, error_detail());
    std::lock_guard lock(registry_mutex);
    return store(std::move(index));
}

PUBLISH(rag, index_add, uint32_t, uint64_t handle, uint32_t dim,
        std::span<const uint8_t> vectors_and_ids) {
    if (dim == 0)
        Mln::raise(Rag::Errors::Index::Dimension::Zero);

    const std::size_t vector_bytes = sizeof(float) * dim;
    const std::size_t stride = vector_bytes + sizeof(uint64_t);
    if (stride == 0 || vectors_and_ids.size() % stride != 0)
        Mln::raise(Rag::Errors::Index::Add::Payload::Invalid, vectors_and_ids.size());

    const uint32_t count = static_cast<uint32_t>(vectors_and_ids.size() / stride);
    if (count == 0)
        return 0;

    std::vector<float> vectors(static_cast<std::size_t>(count) * dim);
    std::vector<uint64_t> ids(count);
    std::memcpy(vectors.data(), vectors_and_ids.data(),
                static_cast<std::size_t>(count) * vector_bytes);
    std::memcpy(ids.data(), vectors_and_ids.data() + static_cast<std::size_t>(count) * vector_bytes,
                count * sizeof(uint64_t));

    std::lock_guard lock(registry_mutex);
    if (tv_add(handle_of(handle), vectors.data(), count, dim, ids.data()) != 0)
        Mln::raise(Rag::Errors::Index::Add::Failed, error_detail());
    return count;
}

PUBLISH(rag, index_remove, uint32_t, uint64_t handle, uint64_t id) {
    std::lock_guard lock(registry_mutex);
    const int32_t removed = tv_remove(handle_of(handle), id);
    if (removed < 0)
        Mln::raise(Rag::Errors::Index::Remove::Failed, error_detail());
    return static_cast<uint32_t>(removed);
}

PUBLISH(rag, index_search, std::vector<uint8_t>, uint64_t handle, uint32_t top_k,
        std::span<const uint8_t> query) {
    if (top_k == 0)
        Mln::raise(Rag::Errors::Index::Search::Limit::Zero);
    if (query.size() % sizeof(float) != 0)
        Mln::raise(Rag::Errors::Index::Search::Query::Invalid, query.size());

    const uint32_t dim = static_cast<uint32_t>(query.size() / sizeof(float));
    std::vector<float> values(dim);
    std::memcpy(values.data(), query.data(), query.size());

    std::vector<float> scores(top_k);
    std::vector<uint64_t> ids(top_k);
    uint32_t count = 0;
    std::lock_guard lock(registry_mutex);
    if (tv_search(handle_of(handle), values.data(), dim, top_k, scores.data(), ids.data(),
                  &count) != 0)
        Mln::raise(Rag::Errors::Index::Search::Failed, error_detail());

    std::vector<uint8_t> out(sizeof(uint32_t) + count * (sizeof(uint64_t) + sizeof(float)));
    std::size_t offset = 0;
    std::memcpy(out.data() + offset, &count, sizeof(count));
    offset += sizeof(count);
    std::memcpy(out.data() + offset, ids.data(), count * sizeof(uint64_t));
    offset += count * sizeof(uint64_t);
    std::memcpy(out.data() + offset, scores.data(), count * sizeof(float));
    return out;
}

PUBLISH(rag, index_serialize, std::vector<uint8_t>, uint64_t handle) {
    std::lock_guard lock(registry_mutex);
    const Tv::Index *index = handle_of(handle);
    const std::size_t needed = tv_to_bytes(index, nullptr, 0);
    if (needed == 0)
        Mln::raise(Rag::Errors::Index::Serialization::Failed, error_detail());

    std::vector<uint8_t> bytes(needed);
    const std::size_t written = tv_to_bytes(index, bytes.data(), bytes.size());
    if (written != needed)
        Mln::raise(Rag::Errors::Index::Serialization::Size::Invalid, written, needed);
    return bytes;
}

PUBLISH(rag, index_stats, std::vector<uint8_t>, uint64_t handle) {
    std::lock_guard lock(registry_mutex);
    const Tv::Index *index = handle_of(handle);
    const uint32_t values[3] = {
        tv_len(index),
        tv_dim(index),
        tv_bit_width(index),
    };
    std::vector<uint8_t> out(sizeof(values));
    std::memcpy(out.data(), values, sizeof(values));
    return out;
}

PUBLISH(rag, index_release, void, uint64_t handle) {
    std::lock_guard lock(registry_mutex);
    (void)handle_of(handle);
    if (persistent.contains(handle))
        return;
    indices.erase(handle);
}
