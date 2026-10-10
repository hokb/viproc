// Vector of trivially copyable values with inline storage for the first N
// elements (no heap allocation for typical array ranks).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace viproc {

template <class T, std::size_t N> class SmallVector {
    static_assert(std::is_trivially_copyable_v<T>, "SmallVector holds trivially copyable values");

  public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    SmallVector() = default;
    SmallVector(size_type n, const T& value) { assign(n, value); }
    explicit SmallVector(size_type n) { assign(n, T{}); }
    SmallVector(std::initializer_list<T> init) { assign(init.begin(), init.end()); }
    template <class It,
              class = std::enable_if_t<!std::is_integral_v<It>>> // not (size, value)
    SmallVector(It first, It last) {
        assign(first, last);
    }
    SmallVector(const SmallVector& other) {
        if (other.size_ <= N) {
            std::memcpy(inline_, other.data_, sizeof(inline_)); // fixed size: inlined
            size_ = other.size_;
        } else {
            assign(other.begin(), other.end());
        }
    }
    SmallVector(SmallVector&& other) noexcept { steal(other); }
    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            if (other.size_ <= N && !on_heap()) {
                std::memcpy(inline_, other.data_, sizeof(inline_));
                size_ = other.size_;
            } else {
                assign(other.begin(), other.end());
            }
        }
        return *this;
    }
    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            release();
            steal(other);
        }
        return *this;
    }
    ~SmallVector() { release(); }

    template <class It> void assign(It first, It last) {
        const auto n = static_cast<size_type>(std::distance(first, last));
        reserve_discard(n);
        std::copy(first, last, data_);
        size_ = n;
    }
    void assign(size_type n, const T& value) {
        reserve_discard(n);
        std::fill_n(data_, n, value);
        size_ = n;
    }

    void resize(size_type n, const T& value = T{}) {
        reserve(n);
        if (n > size_) {
            std::fill(data_ + size_, data_ + n, value);
        }
        size_ = n;
    }
    void reserve(size_type n) {
        if (n <= capacity_) {
            return;
        }
        T* heap = new T[n];
        std::memcpy(heap, data_, size_ * sizeof(T));
        release();
        data_ = heap;
        capacity_ = n;
    }
    void push_back(const T& v) {
        if (size_ == capacity_) {
            const T copy = v; // v may live in this vector
            reserve(capacity_ * 2);
            data_[size_++] = copy;
            return;
        }
        data_[size_++] = v;
    }
    void clear() { size_ = 0; }

    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T* data() { return data_; }
    const T* data() const { return data_; }
    T& operator[](size_type i) { return data_[i]; }
    const T& operator[](size_type i) const { return data_[i]; }
    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }
    iterator begin() { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end() const { return data_ + size_; }

    friend bool operator==(const SmallVector& a, const SmallVector& b) {
        return a.size_ == b.size_ && std::equal(a.begin(), a.end(), b.begin());
    }

  private:
    bool on_heap() const { return data_ != inline_; }
    void release() {
        if (on_heap()) {
            delete[] data_;
            data_ = inline_;
            capacity_ = N;
        }
    }
    // Ensures capacity for n elements; old contents may be dropped.
    void reserve_discard(size_type n) {
        if (n > capacity_) {
            release();
            data_ = new T[n];
            capacity_ = n;
        }
    }
    void steal(SmallVector& other) {
        if (other.on_heap()) {
            data_ = other.data_;
            capacity_ = other.capacity_;
            other.data_ = other.inline_;
            other.capacity_ = N;
        } else {
            data_ = inline_;
            capacity_ = N;
            std::memcpy(inline_, other.inline_, sizeof(inline_));
        }
        size_ = other.size_;
        other.size_ = 0;
    }

    T inline_[N];
    T* data_ = inline_;
    size_type size_ = 0;
    size_type capacity_ = N;
};

// Vector of any movable T with inline storage for the first N elements (for
// per-task operand lists). Growing beyond N moves everything to the heap.
// Only live elements are constructed, moved and destroyed.
template <class T, std::size_t N> class InlineVector {
  public:
    InlineVector() = default;
    InlineVector(InlineVector&& other) noexcept { take(other); }
    InlineVector& operator=(InlineVector&& other) noexcept {
        if (this != &other) {
            clear();
            take(other);
        }
        return *this;
    }
    InlineVector(const InlineVector&) = delete;
    InlineVector& operator=(const InlineVector&) = delete;
    ~InlineVector() { clear(); }

    template <class... Args> T& emplace_back(Args&&... args) {
        if (heap_.empty() && size_ < N) {
            return *std::construct_at(slot(size_++), std::forward<Args>(args)...);
        }
        if (heap_.empty()) {
            heap_.reserve(2 * N);
            for (std::size_t i = 0; i < size_; ++i) {
                heap_.push_back(std::move(*slot(i)));
                std::destroy_at(slot(i));
            }
        }
        T& v = heap_.emplace_back(std::forward<Args>(args)...);
        size_ = heap_.size();
        return v;
    }
    void push_back(T&& v) { emplace_back(std::move(v)); }
    void clear() {
        if (heap_.empty()) {
            std::destroy_n(slot(0), size_);
        }
        heap_.clear();
        size_ = 0;
    }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T* data() { return heap_.empty() ? slot(0) : heap_.data(); }
    const T* data() const { return heap_.empty() ? slot(0) : heap_.data(); }
    T& operator[](std::size_t i) { return data()[i]; }
    const T& operator[](std::size_t i) const { return data()[i]; }
    T* begin() { return data(); }
    T* end() { return data() + size_; }
    const T* begin() const { return data(); }
    const T* end() const { return data() + size_; }

  private:
    T* slot(std::size_t i) { return std::launder(reinterpret_cast<T*>(inline_)) + i; }
    const T* slot(std::size_t i) const {
        return std::launder(reinterpret_cast<const T*>(inline_)) + i;
    }
    void take(InlineVector& other) {
        if (other.heap_.empty()) {
            for (std::size_t i = 0; i < other.size_; ++i) {
                std::construct_at(slot(i), std::move(*other.slot(i)));
            }
            std::destroy_n(other.slot(0), other.size_);
        } else {
            heap_ = std::move(other.heap_);
            other.heap_.clear();
        }
        size_ = std::exchange(other.size_, 0);
    }

    alignas(T) std::byte inline_[N * sizeof(T)];
    std::vector<T> heap_;
    std::size_t size_ = 0;
};

} // namespace viproc
