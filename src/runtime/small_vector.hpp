// Vector of trivially copyable values with inline storage for the first N
// elements (no heap allocation for typical array ranks).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>

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
    SmallVector(const SmallVector& other) { assign(other.begin(), other.end()); }
    SmallVector(SmallVector&& other) noexcept { steal(other); }
    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            assign(other.begin(), other.end());
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
            std::memcpy(inline_, other.inline_, other.size_ * sizeof(T));
        }
        size_ = other.size_;
        other.size_ = 0;
    }

    T inline_[N];
    T* data_ = inline_;
    size_type size_ = 0;
    size_type capacity_ = N;
};

} // namespace viproc
