#include "buffer.hpp"

#include "element_pool.hpp"

namespace viproc {

Buffer::Buffer(std::size_t nbytes) : nbytes_(nbytes) {
    // Inline also for 0 bytes: a buffer never has a null pointer.
    if (nbytes_ <= kInlineBytes) {
        data_.store(inline_, std::memory_order_relaxed);
    } else {
        data_.store(static_cast<std::byte*>(ElementPool::instance().allocate(nbytes_, capacity_)),
                    std::memory_order_relaxed);
    }
}

Buffer::Buffer(std::size_t nbytes, Deferred) : nbytes_(nbytes) {
    if (nbytes_ <= kInlineBytes) {
        data_.store(inline_, std::memory_order_relaxed);
    } else {
        capacity_ = ElementPool::capacity_for(nbytes_);
    }
}

Buffer::Buffer(void* data, std::size_t nbytes, std::shared_ptr<void> owner)
    : data_(static_cast<std::byte*>(data)), nbytes_(nbytes), owner_(std::move(owner)) {}

Buffer::~Buffer() {
    std::byte* p = data_.load(std::memory_order_relaxed);
    if (capacity_ != 0 && p != nullptr) {
        ElementPool::instance().release(p, capacity_);
    }
}

std::byte* Buffer::ensure_allocated() {
    std::byte* p = data_.load(std::memory_order_acquire);
    if (p != nullptr) {
        return p;
    }
    std::size_t capacity = 0;
    auto* block = static_cast<std::byte*>(ElementPool::instance().allocate(nbytes_, capacity));
    if (data_.compare_exchange_strong(p, block, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        return block;
    }
    ElementPool::instance().release(block, capacity); // another thread was first
    return p;
}

bool Buffer::take_block(Buffer& src) {
    std::byte* block = src.data_.load(std::memory_order_acquire);
    if (capacity_ == 0 || src.capacity_ != capacity_ || block == nullptr) {
        return false;
    }
    std::byte* expected = nullptr;
    if (!data_.compare_exchange_strong(expected, block, std::memory_order_acq_rel,
                                       std::memory_order_acquire)) {
        return false; // allocated meanwhile (pinned by the main thread)
    }
    src.data_.store(nullptr, std::memory_order_relaxed);
    return true;
}

} // namespace viproc
