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
