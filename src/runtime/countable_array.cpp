#include "countable_array.hpp"

#include "element_pool.hpp"

namespace viproc {

CountableArray::CountableArray(std::size_t nbytes) : nbytes_(nbytes) {
    // Inline also for 0 bytes: a buffer never has a null pointer.
    if (nbytes_ <= kInlineBytes) {
        handles_[kHost].data.store(inline_, std::memory_order_relaxed);
    } else {
        handles_[kHost].data.store(static_cast<std::byte*>(ElementPool::instance().allocate(
                                       nbytes_, handles_[kHost].capacity)),
                                   std::memory_order_relaxed);
    }
}

CountableArray::CountableArray(std::size_t nbytes, Deferred) : nbytes_(nbytes) {
    if (nbytes_ <= kInlineBytes) {
        handles_[kHost].data.store(inline_, std::memory_order_relaxed);
    } else {
        handles_[kHost].capacity = ElementPool::capacity_for(nbytes_);
    }
}

CountableArray::CountableArray(void* data, std::size_t nbytes, std::shared_ptr<void> owner)
    : nbytes_(nbytes), owner_(std::move(owner)) {
    handles_[kHost].data.store(static_cast<std::byte*>(data), std::memory_order_relaxed);
}

CountableArray::~CountableArray() {
    std::byte* p = handles_[kHost].data.load(std::memory_order_relaxed);
    if (handles_[kHost].capacity != 0 && p != nullptr) {
        ElementPool::instance().release(p, handles_[kHost].capacity);
    }
}

std::byte* CountableArray::ensure_allocated() {
    std::byte* p = handles_[kHost].data.load(std::memory_order_acquire);
    if (p != nullptr) {
        return p;
    }
    std::size_t capacity = 0;
    auto* block = static_cast<std::byte*>(ElementPool::instance().allocate(nbytes_, capacity));
    if (handles_[kHost].data.compare_exchange_strong(p, block, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
        return block;
    }
    ElementPool::instance().release(block, capacity); // another thread was first
    return p;
}

bool CountableArray::take_block(CountableArray& src) {
    std::byte* block = src.handles_[kHost].data.load(std::memory_order_acquire);
    if (handles_[kHost].capacity == 0 || src.handles_[kHost].capacity != handles_[kHost].capacity ||
        block == nullptr) {
        return false;
    }
    std::byte* expected = nullptr;
    if (!handles_[kHost].data.compare_exchange_strong(expected, block, std::memory_order_acq_rel,
                                                      std::memory_order_acquire)) {
        return false; // allocated meanwhile (pinned by the main thread)
    }
    src.handles_[kHost].data.store(nullptr, std::memory_order_relaxed);
    return true;
}

} // namespace viproc
