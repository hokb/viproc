#include "assign.hpp"

#include <cstring>

namespace viproc {

namespace {

class AssignKernel : public Kernel {
  public:
    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& dst = outputs[0].layout;
        const std::size_t item = itemsize(dst.dtype);
        std::vector<Layout> layouts{broadcast_to(inputs[0].layout, dst.shape), dst};
        OperandPointers base{reinterpret_cast<char*>(inputs[0].first()),
                             reinterpret_cast<char*>(outputs[0].first())};
        for_each_row(dst.shape, layouts, std::move(base),
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
        return {};
    }
};

} // namespace

std::shared_ptr<Kernel> make_assign_kernel() { return std::make_shared<AssignKernel>(); }

} // namespace viproc
