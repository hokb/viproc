// Intrusive reference counting for main-thread-only objects (Array, Storage).
//
// Arrays and storages are created, referenced and released by the main thread
// only (workers see Tasks and Buffers), so their counts are plain ints: no
// atomic read-modify-write per copy, unlike std::shared_ptr.
#pragma once

#include <cstddef>
#include <utility>

namespace viproc {

// Base of objects counted by Ref<T>. T provides
// `static void last_ref_dropped(T*)`, called when the count drops to 0.
class MainThreadCounted {
  public:
    MainThreadCounted() = default;
    MainThreadCounted(const MainThreadCounted&) = delete;
    MainThreadCounted& operator=(const MainThreadCounted&) = delete;

    int ref_count() const { return refs_; }

  private:
    template <class T> friend class Ref;
    int refs_ = 0;
};

template <class T> class Ref {
  public:
    Ref() = default;
    Ref(std::nullptr_t) {}
    // Takes a new reference to `p`.
    explicit Ref(T* p) : p_(p) {
        if (p_ != nullptr) {
            ++p_->refs_;
        }
    }
    Ref(const Ref& o) : Ref(o.p_) {}
    Ref(Ref&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    Ref& operator=(Ref o) noexcept {
        std::swap(p_, o.p_);
        return *this;
    }
    ~Ref() { reset(); }

    void reset() {
        T* p = std::exchange(p_, nullptr);
        if (p != nullptr && --p->refs_ == 0) {
            T::last_ref_dropped(p);
        }
    }

    // Hands the reference out as a raw pointer (e.g. across the C ABI);
    // adopt() takes it back.
    T* release() { return std::exchange(p_, nullptr); }
    static Ref adopt(T* p) {
        Ref r;
        r.p_ = p;
        return r;
    }

    T* get() const { return p_; }
    T& operator*() const { return *p_; }
    T* operator->() const { return p_; }
    explicit operator bool() const { return p_ != nullptr; }
    int use_count() const { return p_ != nullptr ? p_->refs_ : 0; }

    friend bool operator==(const Ref& a, const Ref& b) { return a.p_ == b.p_; }

  private:
    T* p_ = nullptr;
};

} // namespace viproc
