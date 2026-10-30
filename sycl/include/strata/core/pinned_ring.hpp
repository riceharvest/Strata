// #1996 (0.1.43.1): host-to-device uploads whose host side is pageable go up through pinned bounce slots.
//
// A cudaMemcpyAsync from pageable memory is staged by the runtime, and the staging is not ordered the way a pinned copy
// is.  On gfx1100 (ROCm 7.2.4) a text prompt after a picture one read stale bytes that way (#1712, fixed for the step
// records and the token ids in 0.1.43); a 4-hour multi-slot load (#1996) still faulted after 31 minutes, with the
// image rows, the draft tokens and bounds and the non-mapped routing tables still going up from caller memory.  The
// analysis of which copies those are, and the idea of a ring of pinned slots with one event each, are
// mantovaniluca91's (patch C on #1712); this is our own form of it.
//
// A PinnedRing owns a few pinned slots, each with an event.  h2d() copies the bytes into the next slot (after the copy
// that last read that slot is done), queues the DMA from the slot and records the slot's event.  So the caller may
// overwrite or free its source the moment the call returns, as a pageable copy allowed.  With no pinned RAM (or
// STRATA_TEST_PAGEABLE=1) the copy is made from the caller's memory and the stream is waited for, so the source is
// free on return all the same.  A ring belongs to ONE device (the one current when it is first used) and is used by one
// thread at a time.
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

class PinnedRing {
public:
    static constexpr int kSlots = 4;

    PinnedRing() = default;
    PinnedRing(const PinnedRing&) = delete;
    PinnedRing& operator=(const PinnedRing&) = delete;
    ~PinnedRing() { release(); }

    // dst (device) <- bytes from src (host, any kind of memory); src is free to change when this returns
    dpct::err0 h2d(void *dst, const void *src, size_t bytes,
                   dpct::queue_ptr s) try {
        if (bytes == 0) return 0;
        void* stage = acquire(bytes);
        if (stage == nullptr) return direct(dst, src, bytes, s);
        std::memcpy(stage, src, bytes);
        return submit(dst, stage, bytes, s);
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    // For a caller that gathers many small pieces into one upload: acquire() hands out a pinned slot of at least
    // `bytes` (nullptr: none could be had - use direct()), the caller fills it, then submit() queues the copy and
    // guards the slot.  An acquired slot must be submitted before the next acquire().
    void *acquire(size_t bytes) try {
        if (bytes == 0 || pageable_forced()) return nullptr;
        int cur = -1;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (DPCT_CHECK_ERROR(cur = dpct::get_current_device_id()) != 0) {
            ; return nullptr;
        }
        if (dev_ < 0) dev_ = cur;
        if (cur != dev_) return nullptr;   // a ring is for one device
        Slot& sl = slots_[next_];
        next_ = (next_ + 1) % kSlots;
        if (sl.pending) {                  // the copy that last read this slot is done before it is refilled
            sl.pending = false;
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            if (DPCT_CHECK_ERROR(sl.ev->wait_and_throw()) != 0) {
                ; return nullptr;
            }
        }
        if (sl.cap < bytes) {
            if (sl.p) {
                sycl::free(sl.p, dpct::get_in_order_queue()); sl.p = nullptr;
                sl.cap = 0;
            } // idle: its last copy was waited for above
            const size_t cap = bytes < kMinCap ? kMinCap : bytes;
            /*
            DPCT1048: The original value cudaHostAllocPortable is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            if (DPCT_CHECK_ERROR(sl.p = (void *)sycl::malloc_host(
                                     cap, dpct::get_in_order_queue())) != 0) {
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                sl.p = nullptr;
                return nullptr;
            }
            sl.cap = cap;
        }
        if (sl.ev == nullptr &&
            DPCT_CHECK_ERROR(sl.ev = new sycl::event()) != 0) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            sl.ev = nullptr;
            return nullptr;
        }
        acquired_ = &sl;
        return sl.p;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    dpct::err0 submit(void *dst, const void *stage, size_t bytes,
                      dpct::queue_ptr s) try {
        Slot* sl = acquired_;
        acquired_ = nullptr;
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        dpct::err0 e = DPCT_CHECK_ERROR(s->memcpy(dst, stage, bytes));

        /*
        DPCT1024: The original code returned the error code that was
        further consumed by the program logic. This original code was replaced
        with 0. You may need to rewrite the program logic consuming the error
        code.
        */
        if (sl != nullptr &&
            DPCT_CHECK_ERROR(dpct::sync_barrier(sl->ev, s)) == 0) {
            sl->pending = true;
            return 0;
        }
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        ; // no event: the copy is waited for here instead
        return DPCT_CHECK_ERROR(s->wait());
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    // the same copy without a pinned slot: from the caller's memory, the stream waited for
    static dpct::err0 direct(void *dst, const void *src, size_t bytes,
                             dpct::queue_ptr s) try {
        if (bytes == 0) return 0;
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        dpct::err0 e = DPCT_CHECK_ERROR(s->memcpy(dst, src, bytes));
        if (e == 0) e = DPCT_CHECK_ERROR(s->wait());
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return e;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    // dst (pageable host) <- device bytes; the stream is drained and the bytes are in dst on return
    dpct::err0 d2h_sync(void *dst, const void *src, size_t bytes,
                        dpct::queue_ptr s) try {
        if (bytes == 0) return 0;
        void* stage = acquire(bytes);
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        dpct::err0 e = DPCT_CHECK_ERROR(
            s->memcpy(stage != nullptr ? stage : dst, src, bytes));
        if (e == 0) e = DPCT_CHECK_ERROR(s->wait());
        if (e == 0 && stage != nullptr) std::memcpy(dst, stage, bytes);
        if (stage != nullptr) acquired_ = nullptr;   // no event: the stream was drained
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return e;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    // Every slot holds at least `bytes` from here on: grown now, by a caller that must not allocate later (a pinned
    // allocation or free takes the driver's lock and can wait for the device: STRATA_PIPED_CONCURRENT grows the rings
    // before a read whose windows run beside it).  On the device that is current, as acquire().  False: a slot could
    // not be had (its uploads then grow it, or go up from the caller's memory, as before).
    bool reserve(size_t bytes) try {
        if (bytes == 0 || pageable_forced()) return true;
        int cur = -1;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (DPCT_CHECK_ERROR(cur = dpct::get_current_device_id()) != 0) {
            ; return false;
        }
        if (dev_ < 0) dev_ = cur;
        if (cur != dev_) return false;
        for (Slot& sl : slots_) {
            if (sl.cap >= bytes && sl.ev != nullptr) continue;
            if (sl.pending) {
                sl.pending = false;
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                if (DPCT_CHECK_ERROR(sl.ev->wait_and_throw()) != 0) {
                    ; return false;
                }
            }
            if (sl.cap < bytes) {
                if (sl.p) {
                    sycl::free(sl.p, dpct::get_in_order_queue());
                    sl.p = nullptr;
                    sl.cap = 0;
                }
                const size_t cap = bytes < kMinCap ? kMinCap : bytes;
                /*
                DPCT1048: The original value cudaHostAllocPortable is not
                meaningful in the migrated code and was removed or replaced with
                0. You may need to check the migrated code.
                */
                if (DPCT_CHECK_ERROR(sl.p = (void *)sycl::malloc_host(
                                         cap, dpct::get_in_order_queue())) !=
                    0) {
                    /*
                    DPCT1026: The call to cudaGetLastError was removed
                    because this functionality is redundant in SYCL.
                    */
                    sl.p = nullptr;
                    return false;
                }
                sl.cap = cap;
            }
            if (sl.ev == nullptr &&
                DPCT_CHECK_ERROR(sl.ev = new sycl::event()) != 0) {
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                sl.ev = nullptr;
                return false;
            }
        }
        return true;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void release() {
        for (Slot& sl : slots_) {
            if (sl.pending && sl.ev) sl.ev->wait_and_throw();
            if (sl.ev) dpct::destroy_event(sl.ev);
            if (sl.p) sycl::free(sl.p, dpct::get_in_order_queue());
            sl = Slot{};
        }
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        acquired_ = nullptr;
        dev_ = -1;
        next_ = 0;
    }

private:
    static constexpr size_t kMinCap = 64 * 1024;
    struct Slot {
        void* p = nullptr;
        size_t cap = 0;
        dpct::event_ptr ev = nullptr;
        bool pending = false;
    };
    static bool pageable_forced() {
        static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
        return v;
    }
    Slot slots_[kSlots];
    Slot* acquired_ = nullptr;
    int next_ = 0;
    int dev_ = -1;
};

}  // namespace strata::core
