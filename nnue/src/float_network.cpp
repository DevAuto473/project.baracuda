#include "barracuda/nnue/float_network.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

namespace barracuda::nnue {

namespace {
constexpr char kMagic[8] = {'B', 'R', 'C', 'D', 'N', 'N', 'U', 'E'};
constexpr std::uint32_t kVersion = 1;

inline float clip01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

template <typename T>
bool write_pod(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
    return bool(f);
}
template <typename T>
bool read_pod(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return bool(f);
}
bool write_vec(std::ofstream& f, const std::vector<float>& v) {
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
    return bool(f);
}
bool read_vec(std::ifstream& f, std::vector<float>& v) {
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
    return bool(f);
}
} // namespace

FloatNetwork::FloatNetwork()
    : ft_w(std::size_t(kInputs) * kHidden), ft_b(kHidden), l1_w(std::size_t(kL1) * 2 * kHidden), l1_b(kL1),
      l2_w(std::size_t(kL2) * kL1), l2_b(kL2), out_w(kL2) {}

void FloatNetwork::init_random(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto fill = [&](std::vector<float>& v, float stddev) {
        std::normal_distribution<float> d(0.0f, stddev);
        for (float& x : v) x = d(rng);
    };
    // ~30 active features: keep the accumulator spread roughly inside the [0,1] ClippedReLU window.
    fill(ft_w, 0.06f);
    std::fill(ft_b.begin(), ft_b.end(), 0.25f);
    fill(l1_w, std::sqrt(2.0f / (2 * kHidden)));
    std::fill(l1_b.begin(), l1_b.end(), 0.05f);
    fill(l2_w, std::sqrt(2.0f / kL1));
    std::fill(l2_b.begin(), l2_b.end(), 0.05f);
    fill(out_w, std::sqrt(1.0f / kL2));
    out_b = 0.0f;
    epochs_trained = 0;
    samples_trained = 0;
    clamp_weights();
}

void FloatNetwork::clamp_weights() {
    auto clampv = [](std::vector<float>& v, float lim) {
        for (float& x : v) x = std::clamp(x, -lim, lim);
    };
    clampv(ft_w, kFtWeightLimit);
    clampv(ft_b, kFtWeightLimit);
    clampv(l1_w, kDenseWeightLimit);
    clampv(l2_w, kDenseWeightLimit);
    clampv(out_w, kDenseWeightLimit);
}

bool FloatNetwork::save(const std::string& path, std::string* error) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        if (error) *error = "cannot open '" + path + "' for writing";
        return false;
    }
    f.write(kMagic, sizeof(kMagic));
    bool ok = write_pod(f, kVersion);
    const std::uint32_t dims[4] = {std::uint32_t(kInputs), std::uint32_t(kHidden), std::uint32_t(kL1), std::uint32_t(kL2)};
    for (std::uint32_t d : dims) ok = ok && write_pod(f, d);
    ok = ok && write_pod(f, epochs_trained) && write_pod(f, samples_trained);
    ok = ok && write_vec(f, ft_w) && write_vec(f, ft_b) && write_vec(f, l1_w) && write_vec(f, l1_b) &&
         write_vec(f, l2_w) && write_vec(f, l2_b) && write_vec(f, out_w) && write_pod(f, out_b);
    if (!ok && error) *error = "write error on '" + path + "'";
    return ok;
}

bool FloatNetwork::load(const std::string& path, std::string* error) {
    auto fail = [&](const std::string& m) {
        if (error) *error = m;
        return false;
    };
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail("cannot open '" + path + "'");
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, kMagic, 8) != 0) return fail("'" + path + "' is not a Barracuda network");
    std::uint32_t version = 0, dims[4] = {};
    if (!read_pod(f, version) || version != kVersion) return fail("unsupported network version");
    for (auto& d : dims)
        if (!read_pod(f, d)) return fail("truncated header");
    if (dims[0] != kInputs || dims[1] != kHidden || dims[2] != kL1 || dims[3] != kL2)
        return fail("network topology mismatch (" + std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x" +
                    std::to_string(dims[2]) + "x" + std::to_string(dims[3]) + ")");
    FloatNetwork tmp;
    bool ok = read_pod(f, tmp.epochs_trained) && read_pod(f, tmp.samples_trained);
    ok = ok && read_vec(f, tmp.ft_w) && read_vec(f, tmp.ft_b) && read_vec(f, tmp.l1_w) && read_vec(f, tmp.l1_b) &&
         read_vec(f, tmp.l2_w) && read_vec(f, tmp.l2_b) && read_vec(f, tmp.out_w) && read_pod(f, tmp.out_b);
    if (!ok) return fail("truncated network file");
    *this = std::move(tmp);
    return true;
}

float FloatNetwork::forward(const int* stm_f, int n_stm, const int* nstm_f, int n_nstm, Trace* trace) const {
    Trace local;
    Trace& t = trace ? *trace : local;

    for (int i = 0; i < kHidden; ++i) t.acc[0][i] = t.acc[1][i] = ft_b[i];
    for (int k = 0; k < n_stm; ++k) {
        const float* w = &ft_w[std::size_t(stm_f[k]) * kHidden];
        for (int i = 0; i < kHidden; ++i) t.acc[0][i] += w[i];
    }
    for (int k = 0; k < n_nstm; ++k) {
        const float* w = &ft_w[std::size_t(nstm_f[k]) * kHidden];
        for (int i = 0; i < kHidden; ++i) t.acc[1][i] += w[i];
    }
    for (int i = 0; i < kHidden; ++i) {
        t.x[i] = clip01(t.acc[0][i]);
        t.x[kHidden + i] = clip01(t.acc[1][i]);
    }
    for (int o = 0; o < kL1; ++o) {
        const float* w = &l1_w[std::size_t(o) * 2 * kHidden];
        float s = l1_b[o];
        for (int i = 0; i < 2 * kHidden; ++i) s += w[i] * t.x[i];
        t.z1[o] = s;
        t.h1[o] = clip01(s);
    }
    for (int o = 0; o < kL2; ++o) {
        const float* w = &l2_w[std::size_t(o) * kL1];
        float s = l2_b[o];
        for (int i = 0; i < kL1; ++i) s += w[i] * t.h1[i];
        t.z2[o] = s;
        t.h2[o] = clip01(s);
    }
    float s = out_b;
    for (int i = 0; i < kL2; ++i) s += out_w[i] * t.h2[i];
    t.out = s;
    return s;
}

} // namespace barracuda::nnue
