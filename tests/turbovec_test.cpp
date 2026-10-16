#include <mln/sdk/native/abi.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {
    int failures = 0;

    bool check(bool condition, const std::string &what) {
        std::printf("%s %s\n", condition ? "ok  " : "FAIL", what.c_str());
        if (!condition)
            ++failures;
        return condition;
    }

    std::map<uint32_t, std::vector<uint8_t>> buffers;
    uint32_t next_buffer = 1;

    uint32_t allocate(void *, uint32_t bytes) {
        const uint32_t id = next_buffer++;
        buffers[id].resize(bytes);
        return id;
    }
    void release(void *, uint32_t id) { buffers.erase(id); }
    uint32_t length(void *, uint32_t id) {
        const auto found = buffers.find(id);
        return found == buffers.end() ? 0 : static_cast<uint32_t>(found->second.size());
    }
    void read(void *, uint32_t id, uint32_t offset, void *out, uint32_t len) {
        std::memcpy(out, buffers.at(id).data() + offset, len);
    }
    void write(void *, uint32_t id, uint32_t offset, const void *in, uint32_t len) {
        std::memcpy(buffers.at(id).data() + offset, in, len);
    }

    const void *data(void *, uint32_t id) { return buffers.at(id).data(); }

    Mln::Sdk::Native::Context context{
        nullptr,
        {allocate, release, length, read, write, data},
    };

    const Mln::Sdk::Native::Module *provider = nullptr;

    const Mln::Sdk::Native::Symbol *symbol(const char *name) {
        for (uint32_t i = 0; i < provider->symbol_count; ++i)
            if (std::strcmp(provider->symbols[i].name, name) == 0)
                return &provider->symbols[i];
        return nullptr;
    }

    uint64_t call(const char *name, std::vector<uint64_t> args, bool has_result = true) {
        const Mln::Sdk::Native::Symbol *target = symbol(name);
        if (!target) {
            check(false, std::string("missing symbol: ") + name);
            return 0;
        }
        uint64_t result = 0;
        Mln::Sdk::Native::Call invocation{
            args.data(),
            static_cast<uint32_t>(args.size()),
            &result,
            has_result ? 1u : 0u,
        };
        target->function(&context, &invocation);
        return result;
    }

    uint32_t buffer_from(const void *data, std::size_t bytes) {
        const uint32_t id = allocate(nullptr, static_cast<uint32_t>(bytes));
        if (bytes)
            write(nullptr, id, 0, data, static_cast<uint32_t>(bytes));
        return id;
    }

    std::vector<float> direction(std::size_t dim, std::size_t index) {
        std::mt19937 generator(static_cast<uint32_t>(0x5bd1e995u + index * 2654435761u));
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> values(dim);
        double norm = 0.0;
        for (float &value : values) {
            value = normal(generator);
            norm += static_cast<double>(value) * value;
        }
        norm = std::sqrt(norm);
        for (float &value : values)
            value = static_cast<float>(value / norm);
        return values;
    }

    struct Result {
            std::vector<uint64_t> ids;
            std::vector<float> scores;
    };

    Result search(uint64_t handle, uint32_t top_k, const std::vector<float> &query) {
        const uint32_t query_buffer = buffer_from(query.data(), query.size() * sizeof(float));
        const uint32_t out =
            static_cast<uint32_t>(call("index_search", {handle, top_k, query_buffer}));

        const std::vector<uint8_t> &bytes = buffers.at(out);
        uint32_t count = 0;
        std::memcpy(&count, bytes.data(), sizeof(count));

        Result result;
        result.ids.resize(count);
        result.scores.resize(count);
        std::memcpy(result.ids.data(), bytes.data() + sizeof(count), count * sizeof(uint64_t));
        std::memcpy(result.scores.data(), bytes.data() + sizeof(count) + count * sizeof(uint64_t),
                    count * sizeof(float));
        return result;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "use: %s <librag_turbovec.so>\n", argv[0]);
        return 2;
    }

    void *library = dlopen(argv[1], RTLD_NOW);
    if (!library) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    provider = reinterpret_cast<const Mln::Sdk::Native::Module *(*)()>(
        dlsym(library, "mln_native_module"))();

    check(std::strcmp(provider->name, "rag") == 0, "provider publica o modulo 'rag'");
    check(provider->abi_version == Mln::Sdk::Native::ABI_VERSION,
          "abi_version bate com a do Runtime");
    check(provider->symbol_count == 8, "os 8 PUBLISH convivem num unico .so");

    for (const char *name : {"index_open", "index_create", "index_add", "index_remove",
                             "index_search", "index_serialize", "index_stats", "index_release"}) {
        const Mln::Sdk::Native::Symbol *found = symbol(name);
        check(found && found->signature != nullptr,
              std::string("rag.") + name + " publica Signature tipada");
    }

    if (const Mln::Sdk::Native::Symbol *found = symbol("index_search")) {
        using Kind = Mln::Sdk::Native::Value::Kind;
        const bool shape =
            found->signature->argument_count == 3 && found->signature->arguments[0] == Kind::i64 &&
            found->signature->arguments[1] == Kind::i32 &&
            found->signature->arguments[2] == Kind::i32 && found->signature->result_count == 1 &&
            found->signature->results[0] == Kind::i32;
        check(shape, "rag.index_search tem a assinatura fisica esperada");
    }

    constexpr uint32_t DIM = 64;
    constexpr uint32_t BITS = 4;
    constexpr uint32_t COUNT = 200;

    const uint64_t handle = call("index_create", {DIM, BITS});
    if (!check(handle != 0, "index_create devolveu handle"))
        return 1;

    std::vector<std::vector<float>> vectors;
    std::vector<uint8_t> payload;
    for (uint32_t i = 0; i < COUNT; ++i) {
        vectors.push_back(direction(DIM, i));
    }
    for (const std::vector<float> &values : vectors) {
        const auto *bytes = reinterpret_cast<const uint8_t *>(values.data());
        payload.insert(payload.end(), bytes, bytes + DIM * sizeof(float));
    }
    for (uint32_t i = 0; i < COUNT; ++i) {
        const uint64_t id = 1000000ULL + i * 37ULL;
        const auto *bytes = reinterpret_cast<const uint8_t *>(&id);
        payload.insert(payload.end(), bytes, bytes + sizeof(uint64_t));
    }

    const uint32_t added = static_cast<uint32_t>(
        call("index_add", {handle, DIM, buffer_from(payload.data(), payload.size())}));
    check(added == COUNT, "index_add inseriu todos os vetores");

    const uint32_t stats_buffer = static_cast<uint32_t>(call("index_stats", {handle}));
    uint32_t stats[3] = {0, 0, 0};
    std::memcpy(stats, buffers.at(stats_buffer).data(), sizeof(stats));
    check(stats[0] == COUNT && stats[1] == DIM && stats[2] == BITS,
          "index_stats reporta len/dim/bit_width corretos");

    std::size_t exact = 0;
    for (uint32_t i = 0; i < COUNT; ++i) {
        const Result found = search(handle, 5, vectors[i]);
        if (!found.ids.empty() && found.ids[0] == 1000000ULL + i * 37ULL)
            ++exact;
    }
    check(exact >= COUNT * 95 / 100, "busca recupera o id exato em >=95% das consultas (" +
                                         std::to_string(exact) + "/" + std::to_string(COUNT) + ")");

    const Result before = search(handle, 8, vectors[0]);
    check(before.ids.size() == 8, "top_k=8 devolve 8 resultados");

    const uint32_t serialized = static_cast<uint32_t>(call("index_serialize", {handle}));
    const std::vector<uint8_t> image = buffers.at(serialized);
    check(!image.empty(), "index_serialize produced a .tvim image");

    const uint64_t reopened = call("index_open", {buffer_from(image.data(), image.size())});
    if (check(reopened != 0, "index_open carregou o .tvim")) {
        const Result after = search(reopened, 8, vectors[0]);
        check(after.ids == before.ids, "os mesmos ids voltam depois do roundtrip");
        check(after.scores == before.scores, "os mesmos scores voltam depois do roundtrip");

        const uint64_t again = call("index_open", {buffer_from(image.data(), image.size())});
        check(again == reopened, "index_open reuses the open index for identical bytes");
    }

    const uint64_t victim = 1000000ULL + 5 * 37ULL;
    check(call("index_remove", {handle, victim}) == 1, "index_remove removes a present id");
    check(call("index_remove", {handle, victim}) == 0,
          "index_remove is idempotent for a missing id");

    const Result without = search(handle, 5, vectors[5]);
    bool gone = true;
    for (uint64_t id : without.ids)
        if (id == victim)
            gone = false;
    check(gone, "the removed id no longer appears in results");

    call("index_release", {handle}, false);

    std::printf("\n%s\n", failures == 0 ? "all tests passed" : "TESTS FAILED");
    return failures == 0 ? 0 : 1;
}
