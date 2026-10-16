#pragma once

#include <cstddef>
#include <cstdint>

namespace Tv {
    struct Index;
}

extern "C" {

Tv::Index *tv_create(uint32_t dim, uint32_t bit_width);
Tv::Index *tv_open(const uint8_t *bytes, size_t len);
void tv_close(Tv::Index *index);

int32_t tv_add(Tv::Index *index, const float *vectors, uint32_t count, uint32_t dim,
               const uint64_t *ids);
int32_t tv_remove(Tv::Index *index, uint64_t id);

int32_t tv_search(const Tv::Index *index, const float *query, uint32_t dim, uint32_t k,
                  float *out_scores, uint64_t *out_ids, uint32_t *out_count);

size_t tv_to_bytes(const Tv::Index *index, uint8_t *out, size_t capacity);

uint32_t tv_len(const Tv::Index *index);
uint32_t tv_dim(const Tv::Index *index);
uint32_t tv_bit_width(const Tv::Index *index);

const char *tv_last_error(void);
}
