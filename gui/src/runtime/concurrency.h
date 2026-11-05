// Barracuda runtime - lock-free primitives used between worker threads and the render thread.
//
//  * TripleBuffer<T>  : single-producer / single-consumer snapshot exchange. The writer never
//                       blocks the reader and vice-versa; the reader always sees the latest
//                       *complete* snapshot (no torn reads, no mutex on the 60 FPS path).
//  * MpmcQueue<T>     : bounded multi-producer/multi-consumer ring (Dmitry Vyukov's algorithm),
//                       used for streaming console log lines from any thread to the GUI.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

namespace barracuda::runtime {

// Fixed 64 bytes: std::hardware_destructive_interference_size triggers ABI warnings on GCC.
inline constexpr std::size_t kCacheLine = 64;

template <typename T>
class TripleBuffer {
public:
    TripleBuffer() = default;
    TripleBuffer(const TripleBuffer&) = delete;
    TripleBuffer& operator=(const TripleBuffer&) = delete;

    // ---- producer side ----
    // The returned buffer may hold an older snapshot: the producer must overwrite it fully.
    T& write_buffer() { return buffers_[back_]; }
    void publish() {
        back_ = middle_.exchange(std::uint8_t(back_ | kDirty), std::memory_order_acq_rel) & kIndexMask;
    }
    // Convenience: copy-assign a complete value and publish it.
    void publish(const T& value) {
        write_buffer() = value;
        publish();
    }

    // ---- consumer side ----
    bool has_new() const { return (middle_.load(std::memory_order_acquire) & kDirty) != 0; }
    // Swaps in the newest snapshot if there is one. Returns true when read_buffer() changed.
    bool update() {
        if (!has_new()) return false;
        front_ = middle_.exchange(front_, std::memory_order_acq_rel) & kIndexMask;
        return true;
    }
    const T& read_buffer() const { return buffers_[front_]; }

private:
    static constexpr std::uint8_t kDirty = 0x4;
    static constexpr std::uint8_t kIndexMask = 0x3;

    T buffers_[3]{};
    alignas(kCacheLine) std::atomic<std::uint8_t> middle_{1};
    alignas(kCacheLine) std::uint8_t back_ = 0;  // producer-owned
    alignas(kCacheLine) std::uint8_t front_ = 2; // consumer-owned
};

template <typename T>
class MpmcQueue {
public:
    explicit MpmcQueue(std::size_t capacity_pow2 = 4096) {
        std::size_t cap = 2;
        while (cap < capacity_pow2) cap <<= 1;
        mask_ = cap - 1;
        cells_ = std::make_unique<Cell[]>(cap);
        for (std::size_t i = 0; i < cap; ++i) cells_[i].seq.store(i, std::memory_order_relaxed);
    }
    MpmcQueue(const MpmcQueue&) = delete;
    MpmcQueue& operator=(const MpmcQueue&) = delete;

    bool try_push(T value) {
        Cell* cell;
        std::size_t pos = enqueue_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &cells_[pos & mask_];
            const std::size_t seq = cell->seq.load(std::memory_order_acquire);
            const std::intptr_t diff = std::intptr_t(seq) - std::intptr_t(pos);
            if (diff == 0) {
                if (enqueue_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
            } else if (diff < 0) {
                return false; // full
            } else {
                pos = enqueue_.load(std::memory_order_relaxed);
            }
        }
        cell->value = std::move(value);
        cell->seq.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out) {
        Cell* cell;
        std::size_t pos = dequeue_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &cells_[pos & mask_];
            const std::size_t seq = cell->seq.load(std::memory_order_acquire);
            const std::intptr_t diff = std::intptr_t(seq) - std::intptr_t(pos + 1);
            if (diff == 0) {
                if (dequeue_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
            } else if (diff < 0) {
                return false; // empty
            } else {
                pos = dequeue_.load(std::memory_order_relaxed);
            }
        }
        out = std::move(cell->value);
        cell->seq.store(pos + mask_ + 1, std::memory_order_release);
        return true;
    }

private:
    struct Cell {
        std::atomic<std::size_t> seq{0};
        T value{};
    };
    std::unique_ptr<Cell[]> cells_;
    std::size_t mask_ = 0;
    alignas(kCacheLine) std::atomic<std::size_t> enqueue_{0};
    alignas(kCacheLine) std::atomic<std::size_t> dequeue_{0};
};

} // namespace barracuda::runtime
