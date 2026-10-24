// sycl/include/strata/core/lean_exchange.hpp - the B70 port: an adaptive VRAM expert tier for the lean verify window.
//
// WHY. With --stream-experts and STRATA_VERIFY_NO_HOST every expert is either in a VRAM cache slot or in the pinned
// host mirror, and the device-built plan reads a mirrored expert over PCIe inside the expert kernels. On a B70 with
// Flash-Next IQ2_XS that is 10-15% of the routed experts and ~100-140 blobs (~1.4 MB each) per verify window: about
// half of a window's time is that PCIe read. The cache is filled from a static profile and never changes (the
// classic adaptive tier is off under NO_HOST: an expert it evicted would be in neither place). The routing of one
// conversation is concentrated: replayed traces show the top 1,024 mirrored experts of a request take ~64% of its
// fetches.
//
// WHAT. An exchange: a mirrored expert P that the windows keep fetching takes the VRAM slot of a cold resident expert V
// of the same layer, and V moves into host memory, so every expert stays in exactly one readable place:
//   1. copy V's slot into a spare pinned host blob B (copy engine, beside the windows: they only read the slot);
//   2. between two windows: the device tables say V lives at B (mirror) and has no slot; then copy P's mirror blob
//      into the slot (no window reads that slot any more);
//   3. between two windows: the device table says P lives in the slot; P's old mirror blob becomes a spare.
// Between two windows nothing reads an expert (the lean window's host waits for the whole graph), so a table patch
// made there and finished before the next window is all the ordering needed. The same bytes are read either way, so
// the outputs do not change. Swaps never touch the slots the prompt path borrows (its loan region), and every swap in
// flight is finished before a prompt is read (drain), so the prompt path's host tables are always whole.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

class ExpertCache;
class GgufExpertSource;

class LeanExchange {
public:
    struct Config {
        int swaps_per_step = 8;     ///< new exchanges started per window at most
        float decay = 0.98f;        ///< the routing scores' decay per window
        float margin = 2.0f;        ///< a candidate must out-score its victim by this much
        float min_score = 2.0f;     ///< and score at least this
        int spares_per_size = 16;   ///< spare pinned host blobs per blob size (bounds the exchanges in flight)
    };
    ~LeanExchange();
    /// `host_res` (n_layers x n_expert, slot or -1) and `d_res` its device copy, `d_mirror` the device plan's mirror
    /// table, `d_usage` the device plan's routing counts (resident_plan_set_mirror). Slots from `loan_first` on are
    /// never touched.
    bool init(const Config& c, ExpertCache* cache, GgufExpertSource* src, int32_t* host_res, int32_t* d_res,
              unsigned long long* d_mirror, const int32_t* d_usage, int64_t n_layers, int64_t n_expert,
              int64_t loan_first, std::string& err);
    bool on() const { return q_ != nullptr; }
    /// Between two windows (nothing reads an expert): move the exchanges in flight on, start new ones.
    bool step(std::string& err);
    /// Finish every exchange in flight (before a prompt is read).
    bool drain(std::string& err);
    int64_t done() const { return done_; }
    int64_t in_flight() const { return (int64_t) fly_.size(); }

private:
    struct Swap {
        int32_t layer = 0, in = 0, out = 0, slot = 0;
        uint8_t* spare = nullptr;
        int phase = 0;   // 1: V's slot copying out; 2: P copying in
        sycl::event ev;
    };
    bool advance(bool wait, std::string& err);
    bool patch(std::string& err);
    void pick();

    Config c_;
    ExpertCache* cache_ = nullptr;
    GgufExpertSource* src_ = nullptr;
    int32_t* host_res_ = nullptr;
    int32_t* d_res_ = nullptr;
    unsigned long long* d_mirror_ = nullptr;
    const int32_t* d_usage_ = nullptr;
    int64_t L_ = 0, NE_ = 0, loan_first_ = 0;
    sycl::queue* q_ = nullptr;        // the copy engine: the slot and mirror copies
    sycl::queue* kq_ = nullptr;       // a compute queue: the table patches
    int32_t* usage_h_ = nullptr;      // pinned: the last snapshot of d_usage
    sycl::event usage_ev_;
    bool usage_live_ = false;
    std::vector<int32_t> usage_prev_;
    std::vector<float> score_;
    std::vector<uint8_t> busy_;       // per (layer, expert): in an exchange in flight
    std::unordered_map<uint64_t, std::vector<uint8_t*>> spares_;   // blob size -> spare pinned blobs
    std::vector<uint8_t*> owned_;     // every spare allocated here (freed at the end; ownership rotates)
    std::vector<Swap> fly_;
    // the table patches of this step: (index, value) pairs, in pinned host memory the patch kernel reads
    int32_t* pr_ = nullptr;           // [2 * cap] res patches
    unsigned long long* pm_ = nullptr;   // [2 * cap] mirror patches
    int npr_ = 0, npm_ = 0, cap_ = 0;
    int64_t done_ = 0;
};

}  // namespace strata::core
