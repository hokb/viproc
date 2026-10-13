#include "buffer.hpp"

#include "element_pool.hpp"

namespace viproc {

Buffer::Buffer(std::size_t nbytes) : nbytes_(nbytes) {
    // Inline also for 0 bytes: a buffer never has a null pointer.
    if (nbytes_ <= kInlineBytes) {
        data_ = inline_;
    } else {
        data_ = static_cast<std::byte*>(ElementPool::instance().allocate(nbytes_, capacity_));
    }
}

Buffer::Buffer(void* data, std::size_t nbytes, std::shared_ptr<void> owner)
    : data_(static_cast<std::byte*>(data)), nbytes_(nbytes), owner_(std::move(owner)) {}

Buffer::~Buffer() {
    if (capacity_ != 0) {
        ElementPool::instance().release(data_, capacity_);
    }
}

} // namespace viproc
