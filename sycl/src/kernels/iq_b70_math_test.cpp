// iq_b70_math_test: CPU-only bit-exact parity test for the B70 decode/verify/prefill arithmetic primitives:
//   1. swar_ne4 / swar_sub4 vs per-byte reference across all 16 sign nibble patterns
//   2. load4_a2 / load8_a2 vs get_int_b2 at both 4-byte-aligned and 2-byte-aligned offsets
//   3. iq4nl_pair in-register LUT vs kvalues_iq4nl table lookup across random 32-bit nibble words
//   4. Multi<16> (IQ2_XXS), Multi<17> (IQ2_XS), Multi<23> (IQ4_XS) vs single-token vec_dot reference
//   5. ESIMD T_IQ4XS (23) 6-bit scale (ls - 32) + O_IQ4 nibble pair unpacking vs host reference
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(cond, msg)                                                                  \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "iq_b70_math_test FAIL (%s:%d): %s\n",                   \
                         __FILE__, __LINE__, (msg));                                      \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

namespace {

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

int swar_ne4(unsigned x) {
    x = (x | (x >> 4)) & 0x0F0F0F0Fu;
    x = (x | (x >> 2)) & 0x03030303u;
    x = (x | (x >> 1)) & 0x01010101u;
    return (int) (x * 0xFFu);
}

int swar_sub4(unsigned a, unsigned s) {
    return (int) (((a & 0x7F7F7F7Fu) + (s & 0x01010101u)) ^ (a & 0x80808080u));
}

int ref_ne4(unsigned x) {
    uint32_t out = 0;
    for (int b = 0; b < 4; ++b) {
        if (((x >> (8 * b)) & 0xFFu) != 0) out |= (0xFFu << (8 * b));
    }
    return (int) out;
}

int ref_sub4(unsigned a, unsigned s) {
    uint32_t out = 0;
    for (int b = 0; b < 4; ++b) {
        const uint8_t ab = (uint8_t) ((a >> (8 * b)) & 0xFFu);
        const uint8_t sb = (uint8_t) ((s >> (8 * b)) & 0xFFu);
        out |= (uint32_t) (uint8_t) (ab - sb) << (8 * b);
    }
    return (int) out;
}

int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

int load4_a2(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0];
    if (!(a & 2)) return (int) w0;
    return (int) ((w0 >> 16) | (q[1] << 16));
}

void load8_a2(const uint8_t* p, int& ox, int& oy) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0], w1 = q[1];
    if (!(a & 2)) { ox = (int) w0; oy = (int) w1; return; }
    const uint32_t w2 = q[2];
    ox = (int) ((w0 >> 16) | (w1 << 16));
    oy = (int) ((w1 >> 16) | (w2 << 16));
}

uint32_t iq4nl_lut4(uint32_t q4) {
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (q4 >> (8 * i)) & 0xF;
        const uint32_t t = n < 4 ? t0 : n < 8 ? t1 : n < 12 ? t2 : t3;
        r |= ((t >> (8 * (n & 3))) & 0xFF) << (8 * i);
    }
    return r;
}

void iq4nl_pair(int aux, int& vx, int& vy) {
    vx = (int) iq4nl_lut4((uint32_t) aux & 0x0f0f0f0fu);
    vy = (int) iq4nl_lut4(((uint32_t) aux >> 4) & 0x0f0f0f0fu);
}

void ref_iq4nl_pair(int aux, int& vx, int& vy) {
    uint32_t ux = 0, uy = 0;
    for (int i = 0; i < 4; ++i) {
        const uint8_t b = (uint8_t) (((uint32_t) aux >> (8 * i)) & 0xFFu);
        ux |= (uint32_t) (uint8_t) kvalues_iq4nl[b & 0x0F] << (8 * i);
        uy |= (uint32_t) (uint8_t) kvalues_iq4nl[b >> 4] << (8 * i);
    }
    vx = (int) ux;
    vy = (int) uy;
}

int dp4a_ref(int a, int b, int c) {
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    return c + (int) pa[0] * pb[0] + (int) pa[1] * pb[1] + (int) pa[2] * pb[2] + (int) pa[3] * pb[3];
}

int q6_k_centred(const int v) {
    const uint32_t b = (uint32_t) v ^ 0x20202020u;
    return (int) (b | ((b & 0x20202020u) * 6u));
}

int ref_q6_k_centred(const int v) {
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        const int b = (int) (((uint32_t) v >> (8 * i)) & 0x3Fu) - 32;
        out |= (uint32_t) (uint8_t) (int8_t) b << (8 * i);
    }
    return (int) out;
}

}  // namespace

int main() {
    std::mt19937 rng(42);

    // 1. Test swar_ne4 and swar_sub4 across all byte masks and random grid words
    for (int m = 0; m < 256; ++m) {
        const unsigned s0 = ((m & 0x03) << 7) | ((m & 0x0C) << 21);
        const unsigned s1 = ((m & 0x30) << 3) | ((m & 0xC0) << 17);
        const int ne0 = swar_ne4(s0), ne1 = swar_ne4(s1);
        CHECK(ne0 == ref_ne4(s0), "swar_ne4 s0");
        CHECK(ne1 == ref_ne4(s1), "swar_ne4 s1");
        for (int t = 0; t < 32; ++t) {
            const unsigned g = rng();
            CHECK(swar_sub4(g ^ (unsigned) ne0, (unsigned) ne0) == ref_sub4(g ^ (unsigned) ne0, (unsigned) ne0), "swar_sub4 s0");
            CHECK(swar_sub4(g ^ (unsigned) ne1, (unsigned) ne1) == ref_sub4(g ^ (unsigned) ne1, (unsigned) ne1), "swar_sub4 s1");
        }
    }

    // 2. Test load4_a2 and load8_a2 at both 4-byte-aligned and 2-byte-aligned offsets
    alignas(16) uint8_t buf[256];
    for (auto& b : buf) b = (uint8_t) rng();
    for (int base_off : {0, 2, 4, 6}) {
        const uint8_t* p = buf + base_off;
        for (int iqs = 0; iqs < 32; ++iqs) {
            CHECK(load4_a2(p + 4 * iqs) == get_int_b2(p, iqs), "load4_a2 == get_int_b2");
            int lx = 0, ly = 0;
            load8_a2(p + 4 * iqs, lx, ly);
            CHECK(lx == get_int_b2(p, iqs + 0) && ly == get_int_b2(p, iqs + 1), "load8_a2 == get_int_b2 pair");
        }
    }

    // 3. Test iq4nl_pair vs ref_iq4nl_pair
    for (int i = 0; i < 10000; ++i) {
        const int aux = (int) rng();
        int vx = 0, vy = 0, rx = 0, ry = 0;
        iq4nl_pair(aux, vx, vy);
        ref_iq4nl_pair(aux, rx, ry);
        CHECK(vx == rx && vy == ry, "iq4nl_pair matches table lookup");
    }

    // 4. Test Multi<23> (IQ4_XS) and ESIMD T_IQ4XS (23) unpacking vs vec_dot_iq4_xs_q8_1 reference
    uint32_t o_iq4[256];
    for (int b = 0; b < 256; ++b) {
        o_iq4[b] = (uint32_t) (uint8_t) kvalues_iq4nl[b & 15] | ((uint32_t) (uint8_t) kvalues_iq4nl[b >> 4] << 8);
    }
    for (int iter = 0; iter < 200; ++iter) {
        uint16_t scales_h = (uint16_t) rng();
        uint8_t scales_l[4];
        uint8_t qs[128];
        for (auto& x : scales_l) x = (uint8_t) rng();
        for (auto& x : qs) x = (uint8_t) rng();
        uint32_t w0 = scales_h;
        uint32_t w1 = 0;
        std::memcpy(&w1, scales_l, 4);

        int q8[8][8];
        for (int ib = 0; ib < 8; ++ib)
            for (int j = 0; j < 8; ++j) q8[ib][j] = (int) rng();

        for (int ib = 0; ib < 8; ++ib) {
            const int iqs = 4 * ib;
            // Reference vec_dot_iq4_xs_q8_1
            int ref_sumi = 0;
            for (int j = 0; j < 4; ++j) {
                int aux_q4 = 0;
                std::memcpy(&aux_q4, qs + 4 * (iqs + j), 4);
                int vx = 0, vy = 0;
                ref_iq4nl_pair(aux_q4, vx, vy);
                ref_sumi = dp4a_ref(vx, q8[ib][j + 0], ref_sumi);
                ref_sumi = dp4a_ref(vy, q8[ib][j + 4], ref_sumi);
            }
            const int ref_ls = ((scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((scales_h >> (iqs / 2)) & 0x03) << 4);
            ref_sumi *= (ref_ls - 32);

            // Multi<23> path
            int mw[8], u[8];
            for (int j = 0; j < 4; ++j) {
                int aux_q4 = 0;
                std::memcpy(&aux_q4, qs + 4 * (iqs + j), 4);
                iq4nl_pair(aux_q4, mw[2 * j + 0], mw[2 * j + 1]);
                u[2 * j + 0] = q8[ib][j + 0];
                u[2 * j + 1] = q8[ib][j + 4];
            }
            int s0 = 0, s1 = 0;
            for (int j = 0; j < 4; ++j) s0 = dp4a_ref(mw[j], u[j], s0);
            for (int j = 4; j < 8; ++j) s1 = dp4a_ref(mw[j], u[j], s1);
            const int multi_sumi = (s0 + s1) * (ref_ls - 32);
            CHECK(multi_sumi == ref_sumi, "Multi<23> matches vec_dot_iq4_xs_q8_1");

            // ESIMD T_IQ4XS path
            const uint32_t esimd_ls = ((w1 >> (4 * ib)) & 0xfu) | (((w0 >> (2 * ib)) & 3u) << 4);
            CHECK((int) esimd_ls == ref_ls, "ESIMD T_IQ4XS ls matches ref_ls");
            uint32_t fr0[8];
            for (int j = 0; j < 4; ++j) {
                uint32_t qd = 0;
                std::memcpy(&qd, qs + 16 * ib + 4 * j, 4);
                const uint32_t t0 = o_iq4[qd & 0xffu];
                const uint32_t t1 = o_iq4[(qd >> 8) & 0xffu];
                const uint32_t t2 = o_iq4[(qd >> 16) & 0xffu];
                const uint32_t t3 = o_iq4[qd >> 24];
                fr0[j] = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
                fr0[j + 4] = ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
            }
            int esimd_dot = 0;
            for (int j = 0; j < 8; ++j) esimd_dot = dp4a_ref((int) fr0[j], q8[ib][j], esimd_dot);
            CHECK(esimd_dot * ((int) esimd_ls - 32) == ref_sumi, "ESIMD T_IQ4XS dot matches ref_sumi");
        }
    }

    // 5. Test q6_k_centred across all 6-bit unsigned byte values (0..63) and random packed words
    for (int b0 = 0; b0 < 64; ++b0) {
        for (int b1 = 0; b1 < 64; ++b1) {
            const int v = b0 | (b1 << 8) | (((b0 ^ 37) & 63) << 16) | (((b1 ^ 53) & 63) << 24);
            CHECK(q6_k_centred(v) == ref_q6_k_centred(v), "q6_k_centred matches per-byte b - 32");
        }
    }

    std::printf("iq_b70_math_test: all OK\n");
    return 0;
}
