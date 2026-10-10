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
    // Inline also for 0 bytes: an allocated buffer never has a null pointer.
    if (nbytes_ <= kInlineBytes) {
        data_ = inline_;
        return;
    }
    data_ = static_cast<std::byte*>(::operator new[](nbytes_, kAlignment));
    owns_ = true;
}

} // namespace viproc
