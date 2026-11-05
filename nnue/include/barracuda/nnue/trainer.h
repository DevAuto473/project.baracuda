// Barracuda NNUE - offline mini-batch trainer (backpropagation + Adam), multi-threaded.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include "barracuda/nnue/dataset.h"
#include "barracuda/nnue/float_network.h"

namespace barracuda::nnue {

struct TrainerConfig {
    int epochs = 10;
    int batch_size = 8192;
    float learning_rate = 1e-3f;
    float lr_decay = 0.9f;          // multiplied into the learning rate after each epoch
    float lambda = 0.75f;           // target = lambda * sigmoid(score) + (1 - lambda) * game result
    float validation_split = 0.03f; // tail fraction of the (shuffled) dataset held out
    int threads = 0;                // 0 = hardware_concurrency - 1 (min 1)
    std::uint64_t seed = 0xB0BACAFE;
};

struct TrainingProgress {
    int epoch = 0;   // 1-based
    int epochs = 0;
    std::uint64_t batch = 0;
    std::uint64_t batches_per_epoch = 0;
    double batch_loss = 0.0;  // MSE in win-probability space
    double train_loss = 0.0;  // running mean for the current epoch
    double val_loss = 0.0;    // valid when epoch_done
    double samples_per_sec = 0.0;
    double learning_rate = 0.0;
    bool epoch_done = false;
};

class Trainer {
public:
    using Callback = std::function<void(const TrainingProgress&)>;

    Trainer(FloatNetwork& net, TrainerConfig cfg);

    // Trains `net` in place. Returns false if `stop` was raised. `data` is shuffled.
    bool run(Dataset& data, const std::atomic<bool>& stop, const Callback& on_batch, const Callback& on_epoch);

    // Mean loss over [begin, end) of `data` (no gradient).
    double loss(const Dataset& data, std::size_t begin, std::size_t end) const;

    static float target(const PackedSample& s, float lambda);

private:
    struct Gradients {
        std::vector<float> ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w;
        float out_b = 0.0f;
        std::vector<std::uint8_t> touched; // FT rows touched in this batch
        std::vector<int> touched_list;
        double loss = 0.0;
        void init();
        void zero();
    };
    struct AdamState {
        std::vector<float> m, v;
    };

    void accumulate(const Dataset& data, const std::uint32_t* idx, std::size_t n, Gradients& g) const;
    void apply(const Gradients& g, std::size_t batch_n);
    void adam(std::vector<float>& w, const float* grad, AdamState& st, float scale, std::size_t offset,
              std::size_t count);

    FloatNetwork& net_;
    TrainerConfig cfg_;
    std::vector<Gradients> grads_;
    AdamState s_ft_w_, s_ft_b_, s_l1_w_, s_l1_b_, s_l2_w_, s_l2_b_, s_out_w_, s_out_b_;
    float lr_ = 1e-3f;
    std::uint64_t step_ = 0;
};

} // namespace barracuda::nnue
