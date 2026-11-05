#include "barracuda/nnue/network.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__AVX2__)
#include <immintrin.h>
#define BARRACUDA_USE_AVX2 1
#elif defined(__SSE4_1__) || defined(BARRACUDA_SSE41)
#include <smmintrin.h>
#define BARRACUDA_USE_SSE41 1
#endif

namespace barracuda::nnue {

const char* simd_backend() {
#if defined(BARRACUDA_USE_AVX2)
    return "AVX2";
#elif defined(BARRACUDA_USE_SSE41)
    return "SSE4.1";
#else
    return "scalar";
#endif
}

namespace {

// ---------------------------------------------------------------------------
// SIMD kernels
// ---------------------------------------------------------------------------

// dst = src + sum(add rows) - sum(sub rows), kHidden int16 lanes.
inline void apply_rows(std::int16_t* dst, const std::int16_t* src, const std::int16_t* const* add, int n_add,
                       const std::int16_t* const* sub, int n_sub) {
#if defined(BARRACUDA_USE_AVX2)
    for (int i = 0; i < kHidden; i += 16) {
        __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
        for (int k = 0; k < n_add; ++k) v = _mm256_add_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i*>(add[k] + i)));
        for (int k = 0; k < n_sub; ++k) v = _mm256_sub_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i*>(sub[k] + i)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
#elif defined(BARRACUDA_USE_SSE41)
    for (int i = 0; i < kHidden; i += 8) {
        __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i));
        for (int k = 0; k < n_add; ++k) v = _mm_add_epi16(v, _mm_load_si128(reinterpret_cast<const __m128i*>(add[k] + i)));
        for (int k = 0; k < n_sub; ++k) v = _mm_sub_epi16(v, _mm_load_si128(reinterpret_cast<const __m128i*>(sub[k] + i)));
        _mm_store_si128(reinterpret_cast<__m128i*>(dst + i), v);
    }
#else
    for (int i = 0; i < kHidden; ++i) {
        int v = src[i];
        for (int k = 0; k < n_add; ++k) v += add[k][i];
        for (int k = 0; k < n_sub; ++k) v -= sub[k][i];
        dst[i] = std::int16_t(v);
    }
#endif
}

// ClippedReLU int16 -> uint8 in [0, 127]
inline void crelu_ft(const std::int16_t* in, std::uint8_t* out) {
#if defined(BARRACUDA_USE_AVX2)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi16(kQA);
    for (int i = 0; i < kHidden; i += 32) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i + 16));
        a = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
        b = _mm256_min_epi16(_mm256_max_epi16(b, zero), qa);
        // packus interleaves 128-bit lanes; permute restores the original order
        __m256i packed = _mm256_packus_epi16(a, b);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), packed);
    }
#else
    for (int i = 0; i < kHidden; ++i) out[i] = std::uint8_t(std::clamp<int>(in[i], 0, kQA));
#endif
}

// sum_i x[i] * w[i]   (x: uint8 in [0,127], w: int8 in [-127,127]); n multiple of 32
inline std::int32_t dot_u8_i8(const std::uint8_t* x, const std::int8_t* w, int n) {
#if defined(BARRACUDA_USE_AVX2)
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        // 127*127*2 = 32258 < 32767 : maddubs can never saturate with our clamps
        const __m256i prod = _mm256_maddubs_epi16(vx, vw);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
#elif defined(BARRACUDA_USE_SSE41)
    const __m128i ones = _mm_set1_epi16(1);
    __m128i acc = _mm_setzero_si128();
    for (int i = 0; i < n; i += 16) {
        const __m128i vx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
        const __m128i vw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(vx, vw), ones));
    }
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0x4E));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0xB1));
    return _mm_cvtsi128_si32(acc);
#else
    std::int32_t s = 0;
    for (int i = 0; i < n; ++i) s += std::int32_t(x[i]) * std::int32_t(w[i]);
    return s;
#endif
}

template <int In, int Out>
inline void dense_crelu(const std::uint8_t* x, const std::int8_t* w, const std::int32_t* b, std::int32_t* raw,
                        std::uint8_t* out) {
    for (int o = 0; o < Out; ++o) {
        const std::int32_t s = b[o] + dot_u8_i8(x, w + o * In, In);
        raw[o] = s;
        out[o] = std::uint8_t(std::clamp(s >> kQBShift, 0, kQA));
    }
}

template <typename T>
T quantise(float v, float scale) {
    const float lo = float(std::numeric_limits<T>::min() + (std::is_same_v<T, std::int8_t> ? 1 : 0));
    const float hi = float(std::numeric_limits<T>::max());
    return T(std::clamp(std::round(v * scale), lo, hi));
}

} // namespace

std::shared_ptr<Network> Network::from_float(const FloatNetwork& f) {
    auto net = std::make_shared<Network>();
    for (std::size_t i = 0; i < f.ft_w.size(); ++i) net->ft_w_[i] = quantise<std::int16_t>(f.ft_w[i], kQA);
    for (int i = 0; i < kHidden; ++i) net->ft_b_[i] = quantise<std::int16_t>(f.ft_b[i], kQA);
    for (std::size_t i = 0; i < f.l1_w.size(); ++i) net->l1_w_[i] = quantise<std::int8_t>(f.l1_w[i], kQB);
    for (int i = 0; i < kL1; ++i) net->l1_b_[i] = std::int32_t(std::lround(f.l1_b[i] * kQA * kQB));
    for (std::size_t i = 0; i < f.l2_w.size(); ++i) net->l2_w_[i] = quantise<std::int8_t>(f.l2_w[i], kQB);
    for (int i = 0; i < kL2; ++i) net->l2_b_[i] = std::int32_t(std::lround(f.l2_b[i] * kQA * kQB));
    for (int i = 0; i < kL2; ++i) net->out_w_[i] = quantise<std::int8_t>(f.out_w[i], kQB);
    net->out_b_ = std::int32_t(std::lround(f.out_b * kQA * kQB));
    net->epochs_trained_ = f.epochs_trained;
    net->samples_trained_ = f.samples_trained;
    return net;
}

void Network::refresh(const Position& pos, Accumulator& acc) const {
    for (Color persp : {WHITE, BLACK}) {
        const std::int16_t* rows[kMaxActive];
        int n = 0;
        Bitboard occ = pos.pieces();
        while (occ && n < kMaxActive) {
            const Square s = pop_lsb(occ);
            rows[n++] = &ft_w_[feature_index(persp, pos.piece_on(s), s) * kHidden];
        }
        apply_rows(acc.v[persp], ft_b_, rows, n, nullptr, 0);
    }
}

void Network::update(const Accumulator& prev, Accumulator& next, const DirtyPiece& dirty) const {
    for (Color persp : {WHITE, BLACK}) {
        const std::int16_t* add[3];
        const std::int16_t* sub[3];
        int na = 0, ns = 0;
        for (int k = 0; k < dirty.count; ++k) {
            if (dirty.from[k] != SQ_NONE) sub[ns++] = &ft_w_[feature_index(persp, dirty.piece[k], dirty.from[k]) * kHidden];
            if (dirty.to[k] != SQ_NONE) add[na++] = &ft_w_[feature_index(persp, dirty.piece[k], dirty.to[k]) * kHidden];
        }
        apply_rows(next.v[persp], prev.v[persp], add, na, sub, ns);
    }
}

void Network::add_feature(Accumulator& acc, Color persp, int feature) const {
    const std::int16_t* row = &ft_w_[feature * kHidden];
    apply_rows(acc.v[persp], acc.v[persp], &row, 1, nullptr, 0);
}

void Network::sub_feature(Accumulator& acc, Color persp, int feature) const {
    const std::int16_t* row = &ft_w_[feature * kHidden];
    apply_rows(acc.v[persp], acc.v[persp], nullptr, 0, &row, 1);
}

int Network::evaluate(const Accumulator& acc, Color stm, Trace* trace) const {
    Trace local;
    Trace& t = trace ? *trace : local;
    crelu_ft(acc.v[stm], t.ft);
    crelu_ft(acc.v[~stm], t.ft + kHidden);
    dense_crelu<2 * kHidden, kL1>(t.ft, l1_w_, l1_b_, t.l1_raw, t.l1);
    dense_crelu<kL1, kL2>(t.l1, l2_w_, l2_b_, t.l2_raw, t.l2);
    std::int32_t s = out_b_;
    for (int i = 0; i < kL2; ++i) s += std::int32_t(t.l2[i]) * out_w_[i];
    t.out_raw = s;
    return int((std::int64_t(s) * kOutputScale) / (kQA * kQB));
}

int Network::evaluate(const Position& pos, Trace* trace) const {
    Accumulator acc;
    refresh(pos, acc);
    return evaluate(acc, pos.side_to_move(), trace);
}

std::shared_ptr<Network> load_network(const std::string& path, std::string* error) {
    FloatNetwork f;
    if (!f.load(path, error)) return nullptr;
    return Network::from_float(f);
}

} // namespace barracuda::nnue
