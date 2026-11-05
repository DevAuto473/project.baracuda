#include "barracuda/nnue/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>

namespace barracuda::nnue {

namespace {
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float dclip(float pre) { return (pre > 0.0f && pre < 1.0f) ? 1.0f : 0.0f; }

constexpr float kBeta1 = 0.9f;
constexpr float kBeta2 = 0.999f;
constexpr float kEps = 1e-8f;
} // namespace

void Trainer::Gradients::init() {
    ft_w.assign(std::size_t(kInputs) * kHidden, 0.0f);
    ft_b.assign(kHidden, 0.0f);
    l1_w.assign(std::size_t(kL1) * 2 * kHidden, 0.0f);
    l1_b.assign(kL1, 0.0f);
    l2_w.assign(std::size_t(kL2) * kL1, 0.0f);
    l2_b.assign(kL2, 0.0f);
    out_w.assign(kL2, 0.0f);
    out_b = 0.0f;
    touched.assign(kInputs, 0);
    touched_list.clear();
    loss = 0.0;
}

void Trainer::Gradients::zero() {
    for (int r : touched_list) {
        std::fill_n(ft_w.begin() + std::ptrdiff_t(r) * kHidden, kHidden, 0.0f);
        touched[r] = 0;
    }
    touched_list.clear();
    std::fill(ft_b.begin(), ft_b.end(), 0.0f);
    std::fill(l1_w.begin(), l1_w.end(), 0.0f);
    std::fill(l1_b.begin(), l1_b.end(), 0.0f);
    std::fill(l2_w.begin(), l2_w.end(), 0.0f);
    std::fill(l2_b.begin(), l2_b.end(), 0.0f);
    std::fill(out_w.begin(), out_w.end(), 0.0f);
    out_b = 0.0f;
    loss = 0.0;
}

Trainer::Trainer(FloatNetwork& net, TrainerConfig cfg) : net_(net), cfg_(cfg), lr_(cfg.learning_rate) {
    auto init = [](AdamState& s, std::size_t n) {
        s.m.assign(n, 0.0f);
        s.v.assign(n, 0.0f);
    };
    init(s_ft_w_, net_.ft_w.size());
    init(s_ft_b_, net_.ft_b.size());
    init(s_l1_w_, net_.l1_w.size());
    init(s_l1_b_, net_.l1_b.size());
    init(s_l2_w_, net_.l2_w.size());
    init(s_l2_b_, net_.l2_b.size());
    init(s_out_w_, net_.out_w.size());
    init(s_out_b_, 1);

    int threads = cfg_.threads;
    if (threads <= 0) threads = std::max(1, int(std::thread::hardware_concurrency()) - 1);
    threads = std::clamp(threads, 1, 64);
    grads_.resize(std::size_t(threads));
    for (auto& g : grads_) g.init();
}

float Trainer::target(const PackedSample& s, float lambda) {
    const float score_stm = s.stm == WHITE ? float(s.score) : -float(s.score);
    const float wdl_white = float(s.result) * 0.5f;
    const float wdl_stm = s.stm == WHITE ? wdl_white : 1.0f - wdl_white;
    return lambda * sigmoid(score_stm / float(kOutputScale)) + (1.0f - lambda) * wdl_stm;
}

void Trainer::accumulate(const Dataset& data, const std::uint32_t* idx, std::size_t n, Gradients& g) const {
    FloatNetwork::Trace t;
    int fs[kMaxActive], fn[kMaxActive];
    float dx[2 * kHidden];
    float dz1[kL1], dz2[kL2];

    for (std::size_t k = 0; k < n; ++k) {
        const PackedSample& s = data[idx[k]];
        const Color stm = Color(s.stm);
        const int ns = s.features(stm, fs);
        const int nn = s.features(~stm, fn);

        const float out = net_.forward(fs, ns, fn, nn, &t);
        const float p = sigmoid(out);
        const float y = target(s, cfg_.lambda);
        const float err = p - y;
        g.loss += double(err) * err;

        // d(loss)/d(out) for loss = (sigmoid(out) - y)^2
        const float dout = 2.0f * err * p * (1.0f - p);

        // output layer
        for (int i = 0; i < kL2; ++i) {
            g.out_w[i] += dout * t.h2[i];
            dz2[i] = dout * net_.out_w[i] * dclip(t.z2[i]);
        }
        g.out_b += dout;

        // L2
        for (int i = 0; i < kL1; ++i) dz1[i] = 0.0f;
        for (int o = 0; o < kL2; ++o) {
            const float d = dz2[o];
            if (d == 0.0f) continue;
            float* gw = &g.l2_w[std::size_t(o) * kL1];
            const float* w = &net_.l2_w[std::size_t(o) * kL1];
            for (int i = 0; i < kL1; ++i) {
                gw[i] += d * t.h1[i];
                dz1[i] += d * w[i];
            }
            g.l2_b[o] += d;
        }
        for (int i = 0; i < kL1; ++i) dz1[i] *= dclip(t.z1[i]);

        // L1
        std::fill(std::begin(dx), std::end(dx), 0.0f);
        for (int o = 0; o < kL1; ++o) {
            const float d = dz1[o];
            if (d == 0.0f) continue;
            float* gw = &g.l1_w[std::size_t(o) * 2 * kHidden];
            const float* w = &net_.l1_w[std::size_t(o) * 2 * kHidden];
            for (int i = 0; i < 2 * kHidden; ++i) {
                gw[i] += d * t.x[i];
                dx[i] += d * w[i];
            }
            g.l1_b[o] += d;
        }

        // Feature transformer (sparse): stm half then nstm half
        for (int i = 0; i < kHidden; ++i) {
            dx[i] *= dclip(t.acc[0][i]);
            dx[kHidden + i] *= dclip(t.acc[1][i]);
            g.ft_b[i] += dx[i] + dx[kHidden + i];
        }
        auto scatter = [&](const int* feats, int nf, const float* d) {
            for (int f = 0; f < nf; ++f) {
                const int row = feats[f];
                if (!g.touched[row]) {
                    g.touched[row] = 1;
                    g.touched_list.push_back(row);
                }
                float* gw = &g.ft_w[std::size_t(row) * kHidden];
                for (int i = 0; i < kHidden; ++i) gw[i] += d[i];
            }
        };
        scatter(fs, ns, dx);
        scatter(fn, nn, dx + kHidden);
    }
}

void Trainer::adam(std::vector<float>& w, const float* grad, AdamState& st, float scale, std::size_t offset,
                   std::size_t count) {
    const float bc1 = 1.0f - std::pow(kBeta1, float(step_));
    const float bc2 = 1.0f - std::pow(kBeta2, float(step_));
    const float step_size = lr_ * std::sqrt(bc2) / bc1;
    for (std::size_t i = offset; i < offset + count; ++i) {
        const float gi = grad[i - offset] * scale;
        st.m[i] = kBeta1 * st.m[i] + (1.0f - kBeta1) * gi;
        st.v[i] = kBeta2 * st.v[i] + (1.0f - kBeta2) * gi * gi;
        w[i] -= step_size * st.m[i] / (std::sqrt(st.v[i]) + kEps);
    }
}

void Trainer::apply(const Gradients& g, std::size_t batch_n) {
    ++step_;
    const float scale = 1.0f / float(batch_n);
    // Dense Adam everywhere except the FT weight matrix, where untouched rows have zero gradient:
    // we still decay their moments so that the optimiser behaves like dense Adam.
    adam(net_.ft_w, g.ft_w.data(), s_ft_w_, scale, 0, net_.ft_w.size());
    adam(net_.ft_b, g.ft_b.data(), s_ft_b_, scale, 0, net_.ft_b.size());
    adam(net_.l1_w, g.l1_w.data(), s_l1_w_, scale, 0, net_.l1_w.size());
    adam(net_.l1_b, g.l1_b.data(), s_l1_b_, scale, 0, net_.l1_b.size());
    adam(net_.l2_w, g.l2_w.data(), s_l2_w_, scale, 0, net_.l2_w.size());
    adam(net_.l2_b, g.l2_b.data(), s_l2_b_, scale, 0, net_.l2_b.size());
    adam(net_.out_w, g.out_w.data(), s_out_w_, scale, 0, net_.out_w.size());
    std::vector<float> ob{net_.out_b};
    adam(ob, &g.out_b, s_out_b_, scale, 0, 1);
    net_.out_b = ob[0];
    net_.clamp_weights();
}

double Trainer::loss(const Dataset& data, std::size_t begin, std::size_t end) const {
    if (end <= begin) return 0.0;
    int fs[kMaxActive], fn[kMaxActive];
    double sum = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        const PackedSample& s = data[i];
        const Color stm = Color(s.stm);
        const int ns = s.features(stm, fs);
        const int nn = s.features(~stm, fn);
        const float p = sigmoid(net_.forward(fs, ns, fn, nn));
        const float e = p - target(s, cfg_.lambda);
        sum += double(e) * e;
    }
    return sum / double(end - begin);
}

bool Trainer::run(Dataset& data, const std::atomic<bool>& stop, const Callback& on_batch, const Callback& on_epoch) {
    if (data.size() < 2) return false;
    data.shuffle(cfg_.seed);

    const std::size_t n_val = std::min<std::size_t>(std::size_t(double(data.size()) * cfg_.validation_split), 200000);
    const std::size_t n_train = data.size() - n_val;
    const std::size_t batch = std::size_t(std::max(1, cfg_.batch_size));
    std::vector<std::uint32_t> order(n_train);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937_64 rng(cfg_.seed ^ 0x9E3779B97F4A7C15ULL);

    TrainingProgress pr;
    pr.epochs = cfg_.epochs;
    pr.batches_per_epoch = (n_train + batch - 1) / batch;

    for (int epoch = 1; epoch <= cfg_.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        pr.epoch = epoch;
        pr.epoch_done = false;
        double epoch_loss = 0.0;
        std::size_t epoch_seen = 0;
        const auto t_epoch = std::chrono::steady_clock::now();

        for (std::size_t b = 0; b < n_train; b += batch) {
            if (stop.load(std::memory_order_relaxed)) return false;
            const std::size_t n = std::min(batch, n_train - b);
            const std::uint32_t* idx = order.data() + b;

            // ---- parallel gradient accumulation ----
            const std::size_t nthreads = std::min<std::size_t>(grads_.size(), std::max<std::size_t>(1, n / 256));
            const std::size_t chunk = (n + nthreads - 1) / nthreads;
            std::vector<std::thread> pool;
            for (std::size_t t = 1; t < nthreads; ++t) {
                const std::size_t lo = t * chunk;
                if (lo >= n) break;
                const std::size_t cnt = std::min(chunk, n - lo);
                pool.emplace_back([this, &data, idx, lo, cnt, t] { accumulate(data, idx + lo, cnt, grads_[t]); });
            }
            accumulate(data, idx, std::min(chunk, n), grads_[0]);
            for (auto& th : pool) th.join();

            // ---- reduce into grads_[0] ----
            Gradients& g0 = grads_[0];
            for (std::size_t t = 1; t < nthreads; ++t) {
                Gradients& gt = grads_[t];
                for (int r : gt.touched_list) {
                    if (!g0.touched[r]) {
                        g0.touched[r] = 1;
                        g0.touched_list.push_back(r);
                    }
                    float* dst = &g0.ft_w[std::size_t(r) * kHidden];
                    const float* src = &gt.ft_w[std::size_t(r) * kHidden];
                    for (int i = 0; i < kHidden; ++i) dst[i] += src[i];
                }
                auto add = [](std::vector<float>& a, const std::vector<float>& c) {
                    for (std::size_t i = 0; i < a.size(); ++i) a[i] += c[i];
                };
                add(g0.ft_b, gt.ft_b);
                add(g0.l1_w, gt.l1_w);
                add(g0.l1_b, gt.l1_b);
                add(g0.l2_w, gt.l2_w);
                add(g0.l2_b, gt.l2_b);
                add(g0.out_w, gt.out_w);
                g0.out_b += gt.out_b;
                g0.loss += gt.loss;
                gt.zero();
            }

            const double batch_loss = g0.loss / double(n);
            apply(g0, n);
            g0.zero();

            epoch_loss += batch_loss * double(n);
            epoch_seen += n;
            net_.samples_trained += n;

            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_epoch).count();
            pr.batch = b / batch + 1;
            pr.batch_loss = batch_loss;
            pr.train_loss = epoch_loss / double(epoch_seen);
            pr.samples_per_sec = secs > 0 ? double(epoch_seen) / secs : 0.0;
            pr.learning_rate = lr_;
            if (on_batch) on_batch(pr);
        }

        pr.val_loss = n_val ? loss(data, n_train, data.size()) : pr.train_loss;
        pr.epoch_done = true;
        net_.epochs_trained++;
        if (on_epoch) on_epoch(pr);
        lr_ *= cfg_.lr_decay;
    }
    return true;
}

} // namespace barracuda::nnue
