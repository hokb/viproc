#include "assign.hpp"

#include <cstring>

namespace viproc {

namespace {

// dst = src element by element; src (dst's dtype) broadcasts to dst's shape.
void copy_elements(const Layout& src, char* s, const Layout& dst, char* d) {
    const std::size_t item = itemsize(dst.dtype);
    std::vector<Layout> layouts{broadcast_to(src, dst.shape), dst};
    for_each_row(dst.shape, layouts, OperandPointers{s, d},
                 [item](char** p, std::intptr_t n, const std::intptr_t* st) {
                     if (st[0] == static_cast<std::intptr_t>(item) && st[1] == st[0]) {
                         std::memmove(p[1], p[0], static_cast<std::size_t>(n) * item);
                         return true;
                     }
                     for (std::intptr_t i = 0; i < n; ++i) {
                         std::memcpy(p[1] + i * st[1], p[0] + i * st[0], item);
                     }
                     return true;
                 });
}

class AssignKernel : public Kernel {
  public:
    const char* name() const override { return "assignment (a[...] = v)"; }
    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        copy_elements(inputs[0].layout(), reinterpret_cast<char*>(inputs[0].first()),
                      outputs[0].layout(), reinterpret_cast<char*>(outputs[0].first()));
        return {};
    }
};

class PlaceKernel : public Kernel {
  public:
    explicit PlaceKernel(std::vector<Part> parts) : parts_(std::move(parts)) {}
    const char* name() const override { return "concatenate (copies into parts)"; }
    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& out = outputs[0].layout();
        char* base = reinterpret_cast<char*>(outputs[0].first());
        for (std::size_t i = 0; i < parts_.size(); ++i) {
            const Part& p = parts_[i];
            const Layout dst{out.dtype, p.shape, p.strides, 0};
            copy_elements(inputs[i].layout(), reinterpret_cast<char*>(inputs[i].first()), dst,
                          base + p.offset);
        }
        return {};
    }

  private:
    std::vector<Part> parts_;
};

} // namespace

std::shared_ptr<Kernel> make_assign_kernel() { return std::make_shared<AssignKernel>(); }

std::shared_ptr<Kernel> make_place_kernel(std::vector<Part> parts) {
    return std::make_shared<PlaceKernel>(std::move(parts));
}

} // namespace viproc

namespace viproc {

namespace {

class CreateKernel : public Kernel {
  public:
    CreateKernel(std::vector<std::int64_t> dims, char order, std::vector<std::byte> fill, bool iota,
                 const char* name)
        : dims_(std::move(dims)), order_(order), fill_(std::move(fill)), iota_(iota), name_(name) {}

    const char* name() const override { return name_; }
    bool sizes_from_data() const override { return true; }

    KernelResult size_from_data(std::span<const Operand> inputs,
                                std::span<Layout> outputs) override {
        Shape shape;
        std::size_t k = 0;
        for (const std::int64_t d : dims_) {
            std::int64_t v = d;
            if (v < 0) {
                const Operand& in = inputs[k++];
                v = in.layout().dtype == DType::Int32
                        ? *reinterpret_cast<const std::int32_t*>(in.first())
                        : *reinterpret_cast<const std::int64_t*>(in.first());
                if (iota_) {
                    v = std::max<std::int64_t>(v, 0); // arange(stop <= 0) is empty
                } else if (v < 0) {
                    return KernelResult{1, "ValueError: negative dimensions are not allowed", 0};
                }
            }
            shape.push_back(v);
        }
        outputs[0].shape = std::move(shape);
        return {};
    }

    void layout(std::span<const Layout* const> inputs, std::span<Layout> outputs) override {
        (void)inputs;
        Layout& o = outputs[0];
        o.offset = 0;
        o.strides.resize(o.shape.size());
        std::int64_t step = static_cast<std::int64_t>(itemsize(o.dtype));
        if (order_ == 'F') {
            for (std::size_t i = 0; i < o.shape.size(); ++i) {
                o.strides[i] = step;
                step *= o.shape[i];
            }
        } else {
            for (std::size_t i = o.shape.size(); i-- > 0;) {
                o.strides[i] = step;
                step *= o.shape[i];
            }
        }
    }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        (void)inputs;
        const Layout& l = outputs[0].layout();
        const std::int64_t n = l.size();
        std::byte* p = outputs[0].first(); // dense: C or F order
        if (iota_) {
            auto* v = reinterpret_cast<std::int64_t*>(p);
            for (std::int64_t i = 0; i < n; ++i) {
                v[i] = i;
            }
            return {};
        }
        const std::size_t is = itemsize(l.dtype);
        bool zero = true;
        for (const std::byte b : fill_) {
            zero = zero && b == std::byte{0};
        }
        if (zero) {
            std::memset(p, 0, static_cast<std::size_t>(n) * is);
            return {};
        }
        for (std::int64_t i = 0; i < n; ++i) {
            std::memcpy(p + static_cast<std::size_t>(i) * is, fill_.data(), is);
        }
        return {};
    }

  private:
    std::vector<std::int64_t> dims_;
    char order_;
    std::vector<std::byte> fill_;
    bool iota_;
    const char* name_;
};

} // namespace

std::shared_ptr<Kernel> make_create_kernel(std::vector<std::int64_t> dims, char order,
                                           std::vector<std::byte> fill, bool iota,
                                           const char* name) {
    return std::make_shared<CreateKernel>(std::move(dims), order, std::move(fill), iota, name);
}

} // namespace viproc
