// sycl/src/core/lean_exchange.cpp - see strata/core/lean_exchange.hpp
#include "strata/core/lean_exchange.hpp"

#include "strata/copy_queue.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/gguf_expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <dpct/dpct.hpp>

#include <algorithm>
#include <cstdio>

namespace strata::core {

namespace {
bool done_ev(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}
uint64_t blob_of(int64_t layer) { return strata::kernels::cpu::expert_layout().blob_bytes(layer); }
}  // namespace

LeanExchange::~LeanExchange() {
    if (q_ != nullptr) {
        try {
            q_->wait();
            if (kq_) kq_->wait();
            const sycl::context ctx = kq_->get_context();
            for (uint8_t* p : owned_) sycl::free(p, ctx);
            if (usage_h_) sycl::free(usage_h_, ctx);
            if (pr_) sycl::free(pr_, ctx);
            if (pm_) sycl::free(pm_, ctx);
        } catch (...) {}
    }
}

bool LeanExchange::init(const Config& c, ExpertCache* cache, GgufExpertSource* src, int32_t* host_res, int32_t* d_res,
                        unsigned long long* d_mirror, const int32_t* d_usage, int64_t n_layers, int64_t n_expert,
                        int64_t loan_first, std::string& err) {
    c_ = c;
    cache_ = cache; src_ = src; host_res_ = host_res; d_res_ = d_res; d_mirror_ = d_mirror; d_usage_ = d_usage;
    L_ = n_layers; NE_ = n_expert; loan_first_ = loan_first;
    if (!cache || !src || !host_res || !d_res || !d_mirror || !d_usage || L_ <= 0 || NE_ <= 0) {
        err = "lean exchange: missing tables";
        return false;
    }
    sycl::queue& dq = dpct::get_in_order_queue();
    kq_ = new sycl::queue(dq.get_context(), dq.get_device(), sycl::property::queue::in_order());
    sycl::queue* ce = strata::make_copy_engine_queue("the lean exchange's expert");
    sycl::queue* q = ce != nullptr ? ce : new sycl::queue(dq.get_context(), dq.get_device(), sycl::property::queue::in_order());
    const size_t n = (size_t) (L_ * NE_);
    usage_h_ = sycl::malloc_host<int32_t>(n, *kq_);
    usage_prev_.assign(n, 0);
    score_.assign(n, 0.0f);
    busy_.assign(n, 0);
    int total = 0;
    for (int64_t l = 0; l < L_; ++l) {
        const uint64_t b = blob_of(l);
        if (spares_.count(b)) continue;
        auto& v = spares_[b];
        for (int i = 0; i < c_.spares_per_size; ++i) {
            uint8_t* p = sycl::malloc_host<uint8_t>(b, *kq_);
            if (p == nullptr) { err = "lean exchange: no pinned memory for the spares"; return false; }
            v.push_back(p);
            owned_.push_back(p);
            ++total;
        }
    }
    cap_ = 2 * total + 8;
    pr_ = sycl::malloc_host<int32_t>((size_t) cap_ * 2, *kq_);
    pm_ = sycl::malloc_host<unsigned long long>((size_t) cap_ * 2, *kq_);
    if (!usage_h_ || !pr_ || !pm_) { err = "lean exchange: no pinned memory"; return false; }
    q_ = q;
    std::fprintf(stderr, "strata: lean exchange on: up to %d swaps a window, %d spare blobs (%zu sizes), the copies on %s; "
                         "slots from %lld (the prompt's loan) are left alone\n", c_.swaps_per_step, total, spares_.size(),
                 ce != nullptr ? "the copy engine" : "a compute queue", (long long) loan_first_);
    return true;
}

bool LeanExchange::patch(std::string& err) {
    if (npr_ == 0 && npm_ == 0) return true;
    int32_t* res = d_res_;
    unsigned long long* mir = d_mirror_;
    const int32_t* pr = pr_;
    const unsigned long long* pm = pm_;
    const int nr = npr_, nm = npm_;
    try {
        kq_->single_task<class lean_exchange_patch>([=]() {
            for (int i = 0; i < nr; ++i) res[pr[2 * i]] = pr[2 * i + 1];
            for (int i = 0; i < nm; ++i) mir[pm[2 * i]] = pm[2 * i + 1];
        }).wait();
    } catch (const std::exception& e) {
        err = std::string("lean exchange: table patch: ") + e.what();
        return false;
    }
    npr_ = npm_ = 0;
    return true;
}

bool LeanExchange::advance(bool wait, std::string& err) {
    auto put_res = [&](int64_t i, int32_t v) { pr_[2 * npr_] = (int32_t) i; pr_[2 * npr_ + 1] = v; ++npr_; };
    auto put_mir = [&](int64_t i, unsigned long long v) { pm_[2 * npm_] = (unsigned long long) i; pm_[2 * npm_ + 1] = v; ++npm_; };
    for (size_t k = 0; k < fly_.size();) {
        Swap& s = fly_[k];
        const int64_t iv = s.layer * NE_ + s.out, ip = s.layer * NE_ + s.in;
        if (s.phase == 1) {
            if (wait) s.ev.wait_and_throw();
            else if (!done_ev(s.ev)) { ++k; continue; }
            // V is in the spare now: from the next window on it is read from there, and its slot is free
            put_res(iv, kNotResident);
            put_mir(iv, (unsigned long long) s.spare);
            host_res_[iv] = kNotResident;
            src_->set_mirror_blob(s.layer, s.out, s.spare);
            const uint8_t* from = src_->mirror_blob(s.layer, s.in);
            if (from == nullptr) { err = "lean exchange: the promoted expert lost its mirror blob"; return false; }
            s.ev = q_->memcpy(cache_->device_slot(s.slot), from, (size_t) blob_of(s.layer));
            s.phase = 2;
            if (!wait) { ++k; continue; }
        }
        if (s.phase == 2) {
            if (wait) {
                if (!patch(err)) return false;   // V's entries first: the slot is no longer V's before P's lands
                s.ev.wait_and_throw();
            } else if (!done_ev(s.ev)) { ++k; continue; }
            put_res(ip, s.slot);
            put_mir(ip, 0ull);
            host_res_[ip] = s.slot;
            cache_->replace(s.layer, s.out, s.in);
            uint8_t* freed = src_->mirror_blob(s.layer, s.in);
            src_->set_mirror_blob(s.layer, s.in, nullptr);
            spares_[blob_of(s.layer)].push_back(freed);   // nothing reads it after this step's patch
            busy_[(size_t) iv] = busy_[(size_t) ip] = 0;
            ++done_;
            fly_[k] = fly_.back();
            fly_.pop_back();
            continue;
        }
        ++k;
    }
    return true;
}

void LeanExchange::pick() {
    std::vector<std::pair<float, int32_t>> cand;
    const size_t n = (size_t) (L_ * NE_);
    for (size_t i = 0; i < n; ++i)
        if (host_res_[i] < 0 && !busy_[i] && score_[i] >= c_.min_score &&
            src_->mirror_blob((int64_t) i / NE_, (int64_t) i % NE_) != nullptr)
            cand.emplace_back(score_[i], (int32_t) i);
    if (cand.empty()) return;
    const size_t top = std::min(cand.size(), (size_t) c_.swaps_per_step * 4);
    std::partial_sort(cand.begin(), cand.begin() + (ptrdiff_t) top, cand.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    int started = 0;
    for (size_t k = 0; k < top && started < c_.swaps_per_step; ++k) {
        const int64_t ip = cand[k].second, l = ip / NE_;
        auto& sp = spares_[blob_of(l)];
        if (sp.empty()) continue;
        int64_t iv = -1;
        float best = 0.0f;
        for (int64_t e = 0; e < NE_; ++e) {
            const int64_t i = l * NE_ + e;
            const int32_t sl = host_res_[i];
            if (sl < 0 || sl >= loan_first_ || busy_[(size_t) i]) continue;
            if (iv < 0 || score_[(size_t) i] < best) { iv = i; best = score_[(size_t) i]; }
        }
        if (iv < 0 || cand[k].first < best + c_.margin) continue;
        Swap s;
        s.layer = (int32_t) l; s.in = (int32_t) (ip % NE_); s.out = (int32_t) (iv % NE_); s.slot = host_res_[iv];
        s.spare = sp.back();
        sp.pop_back();
        s.ev = q_->memcpy(s.spare, cache_->device_slot(s.slot), (size_t) blob_of(l));
        s.phase = 1;
        busy_[(size_t) ip] = busy_[(size_t) iv] = 1;
        fly_.push_back(s);
        ++started;
    }
}

bool LeanExchange::step(std::string& err) {
    if (q_ == nullptr) return true;
    try {
        if (usage_live_ && done_ev(usage_ev_)) {   // the routing since the last snapshot
            const size_t n = (size_t) (L_ * NE_);
            for (size_t i = 0; i < n; ++i) {
                const int32_t d = usage_h_[i] - usage_prev_[i];
                usage_prev_[i] = usage_h_[i];
                score_[i] = score_[i] * c_.decay + (float) d;
            }
            usage_live_ = false;
        }
        if (!usage_live_) {
            usage_ev_ = q_->memcpy(usage_h_, d_usage_, (size_t) (L_ * NE_) * sizeof(int32_t));
            usage_live_ = true;
        }
        if (!advance(false, err)) return false;
        pick();
        return patch(err);
    } catch (const std::exception& e) {
        err = std::string("lean exchange: ") + e.what();
        return false;
    }
}

bool LeanExchange::drain(std::string& err) {
    if (q_ == nullptr) return true;
    try {
        while (!fly_.empty())
            if (!advance(true, err)) return false;
        if (usage_live_) { usage_ev_.wait_and_throw(); }
        return patch(err);
    } catch (const std::exception& e) {
        err = std::string("lean exchange: ") + e.what();
        return false;
    }
}

}  // namespace strata::core
