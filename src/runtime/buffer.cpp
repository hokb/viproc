#include "buffer.hpp"

#include <new>

namespace viproc {

namespace {
constexpr std::align_val_t kAlignment{64};
}

Buffer::Buffer(std::size_t nbytes) : nbytes_(nbytes) {}

Buffer::Buffer(void* data, std::size_t nbytes, std::shared_ptr<void> owner)
    : data_(static_cast<std::byte*>(data)), nbytes_(nbytes), owner_(std::move(owner)) {}

Buffer::~Buffer() {
    if (owns_) {
        ::operator delete[](data_, kAlignment);
    }
}

void Buffer::ensure_allocated() {
    if (data_ != nullptr) {
        return;
    }
    // At least one byte so that an allocated buffer never has a null pointer.
    data_ = static_cast<std::byte*>(::operator new[](nbytes_ > 0 ? nbytes_ : 1, kAlignment));
    owns_ = true;
}

} // namespace viproc
