// SYCL port: the prompt path's QSA block scores on XMX (ESIMD DPAS) - see qsa_select_xmx.dp.cpp.
#pragma once
#include "strata/kernels/qsa.hpp"
#include <cstdint>
namespace strata::kernels {
/// The scorer's scratch for a chunk: the pooled keys of blocks [0, key_blocks) and up to max_q queries, split into FP16
/// hi + lo halves (512 bytes a row) with a power-of-two scale per row.
struct QsaSelectXmx {
    void* scratch = nullptr;
    int64_t key_blocks = 0, max_q = 0;
};
/// Bytes of scratch a QsaSelectXmx of key_blocks blocks and max_q queries needs.
uint64_t qsa_select_xmx_bytes(int64_t key_blocks, int64_t max_q);
/// Once per layer and chunk, after the indexer appends: split pooled[0, x.key_blocks) into the scratch. False (nothing
/// launched) when the device has no XMX scorer (not Xe2) or the shapes are not the compiled geometry.
bool qsa_select_xmx_keys(const QsaSelectXmx& x, const float* pooled, const QsaShapes& s, void* stream);
/// qsa_block_scores' contract for a batch of nq <= x.max_q queries whose blocks are all below x.key_blocks: scores[q][b]
/// for b < n_bid (relu per indexer head, summed in head order) and the tail block n_bid (the `dead` key, +1e9 when it
/// has cells). FP32-level (FP16 hi * hi + hi * lo + lo * hi products, FP32 sums), not bitwise.
bool qsa_block_scores_xmx(const QsaSelectXmx& x, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                          int64_t max_blocks, int64_t active_blocks, float* scores, void* stream);
}
