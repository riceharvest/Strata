// gguf_expert_source_test: CPU-only unit test for host mirror hugepage advisory and buddyinfo/THP/smaps parsers (#3681).
#include "strata/core/gguf_expert_source.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#endif

using namespace strata::core;

namespace strata::core {
bool ExpertSource::copy_blob(int64_t, int64_t, uint8_t*) { return false; }
}

#define CHECK(cond, msg)                                                                  \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "gguf_expert_source_test FAIL (%s:%d): %s\n",            \
                         __FILE__, __LINE__, (msg));                                      \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

int main() {
    // 1. parse_buddyinfo: multi-node Normal zone summation, ignoring DMA and DMA32
    const char* sample_buddy =
        "Node 0, zone      DMA      0      0      0      0      2      1      1      0      1      1      3\n"
        "Node 0, zone    DMA32   1200    850    410    200    110     55     22     10      5      2      8\n"
        "Node 0, zone   Normal  48120  21340   8412   3100   1024    412    128     42     19     30     45\n"
        "Node 1, zone   Normal  10000   5000   2000   1000    500    200    100     50     25     12     18\n";
    const BuddyInfoStats b = parse_buddyinfo(sample_buddy);
    CHECK(b.valid, "buddyinfo valid");
    CHECK(b.order9 == 42, "buddyinfo order9 == 30 + 12");
    CHECK(b.order10 == 63, "buddyinfo order10 == 45 + 18");
    CHECK(b.high_order_blocks == 105, "buddyinfo high_order_blocks == 105");
    CHECK(b.high_order_bytes == (42ull * (2ull << 20) + 63ull * (4ull << 20)), "buddyinfo high_order_bytes");

    const char* frag_buddy =
        "Node 0, zone   Normal 992104 412001 112040  18400   2100    180     12      1      0      3      2\n";
    const BuddyInfoStats bf = parse_buddyinfo(frag_buddy);
    CHECK(bf.valid && bf.high_order_blocks == 5 && bf.high_order_blocks < 64, "fragmented buddyinfo detected (< 64)");

    // 2. parse_thp_mode: active bracketed token extraction
    CHECK(parse_thp_mode("always [madvise] never\n") == "madvise", "THP madvise");
    CHECK(parse_thp_mode("[always] madvise never\n") == "always", "THP always");
    CHECK(parse_thp_mode("always madvise [never]\n") == "never", "THP never");
    CHECK(parse_thp_mode("malformed") == "", "THP malformed");

    // 3. parse_anon_hugepages_kb: /proc/self/smaps_rollup format
    const char* sample_smaps =
        "55a000000000-7fffffffffff ---p 00000000 00:00 0                          [rollup]\n"
        "Rss:            27451392 kB\n"
        "Pss:            27449100 kB\n"
        "AnonHugePages:  26843520 kB\n"
        "Shared_Clean:       2292 kB\n";
    CHECK(parse_anon_hugepages_kb(sample_smaps) == 26843520ull, "parse_anon_hugepages_kb");

    // 4. advise_mirror_hugepage on an anonymous mapping + STRATA_MIRROR_HUGEPAGE=0 override
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    const size_t sz = 4ull << 20;
    void* p = ::mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(p != MAP_FAILED, "mmap 4 MiB");
    ::unsetenv("STRATA_MIRROR_HUGEPAGE");
    CHECK(advise_mirror_hugepage(static_cast<uint8_t*>(p) + 128, sz - 256), "advise_mirror_hugepage enabled");
    ::setenv("STRATA_MIRROR_HUGEPAGE", "0", 1);
    CHECK(!advise_mirror_hugepage(p, sz), "advise_mirror_hugepage disabled via env");
    ::unsetenv("STRATA_MIRROR_HUGEPAGE");
    ::munmap(p, sz);
#endif

    std::printf("gguf_expert_source_test: all OK\n");
    return 0;
}
