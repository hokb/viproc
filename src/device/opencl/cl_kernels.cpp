#include "device/opencl/cl_kernels.hpp"

#include "device/opencl/cl_fft.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstring>
#include <new>
#include <vector>

namespace viproc::opencl {

namespace {

// Matrices up to this order are factored by one work-group each, unblocked
// (many small matrices: one launch for the whole batch); larger ones blocked
// with panels of kPanel columns and CLBlast's trsm/gemm/syrk/herk for the
// trailing updates.
constexpr int kSmallN = 64;
constexpr int kPanel = 64;

using Bytes = std::vector<std::byte>;

// Copies an n-d block of elements of `es` bytes between two strided layouts.
void copy_nd(std::size_t nd, const std::int64_t* shape, const std::byte* src,
             const std::int64_t* sstr, std::byte* dst, const std::int64_t* dstr, std::size_t es) {
    for (std::size_t a = 0; a < nd; ++a) {
        if (shape[a] == 0) {
            return;
        }
    }
    if (nd == 0) {
        std::memcpy(dst, src, es);
        return;
    }
    const std::int64_t n = shape[nd - 1], ss = sstr[nd - 1], ds = dstr[nd - 1];
    std::vector<std::int64_t> idx(nd, 0);
    for (;;) {
        std::int64_t so = 0, d0 = 0;
        for (std::size_t a = 0; a + 1 < nd; ++a) {
            so += idx[a] * sstr[a];
            d0 += idx[a] * dstr[a];
        }
        const std::byte* s = src + so;
        std::byte* d = dst + d0;
        switch (es) {
        case 4:
            for (std::int64_t i = 0; i < n; ++i) {
                std::memcpy(d + i * ds, s + i * ss, 4);
            }
            break;
        case 8:
            for (std::int64_t i = 0; i < n; ++i) {
                std::memcpy(d + i * ds, s + i * ss, 8);
            }
            break;
        case 16:
            for (std::int64_t i = 0; i < n; ++i) {
                std::memcpy(d + i * ds, s + i * ss, 16);
            }
            break;
        default:
            for (std::int64_t i = 0; i < n; ++i) {
                std::memcpy(d + i * ds, s + i * ss, es);
            }
        }
        std::size_t a = nd - 1;
        for (;;) {
            if (a == 0) {
                return;
            }
            --a;
            if (++idx[a] < shape[a]) {
                break;
            }
            idx[a] = 0;
        }
    }
}

std::vector<std::int64_t> dense_strides(const std::vector<std::int64_t>& shape, std::size_t es) {
    std::vector<std::int64_t> s(shape.size());
    std::int64_t step = static_cast<std::int64_t>(es);
    for (std::size_t a = shape.size(); a-- > 0;) {
        s[a] = step;
        step *= std::max<std::int64_t>(shape[a], 1);
    }
    return s;
}

// One operand of a gufunc op: its first element, its strides along the
// outputs' batch axes (0 where it broadcasts) and its core axes.
struct View {
    std::byte* base = nullptr;
    std::vector<std::int64_t> bstr;
    std::vector<int> labels; // present core labels, in axis order
    std::vector<std::int64_t> cshape, cstr;
};

struct Plan {
    std::vector<std::int64_t> batch;
    std::int64_t count = 1;          // batch elements
    std::vector<std::int64_t> sizes; // per label (1 for a missing one)
    std::vector<View> ops;           // inputs, then outputs
};

Plan make_plan(const numpy::GufuncCore& core, std::size_t nin, std::span<const Operand> inputs,
               std::span<const Operand> outputs) {
    Plan p;
    const std::size_t nop = core.axes.size();
    p.sizes.assign(static_cast<std::size_t>(core.nlabels), 1);
    const Layout& o = outputs[0].layout();
    const std::size_t nb = o.shape.size() - core.present(nin);
    p.batch.assign(o.shape.begin(), o.shape.begin() + static_cast<std::ptrdiff_t>(nb));
    for (std::int64_t e : p.batch) {
        p.count *= e;
    }
    for (std::size_t i = 0; i < nop; ++i) {
        const Operand& op = i < nin ? inputs[i] : outputs[i - nin];
        const Layout& l = op.layout();
        View v;
        v.base = op.first();
        const std::size_t lb = l.shape.size() - core.present(i);
        v.bstr.assign(nb, 0);
        for (std::size_t a = 0; a < lb; ++a) {
            if (l.shape[a] != 1) {
                v.bstr[nb - lb + a] = l.strides[a];
            }
        }
        std::size_t ax = lb;
        for (int label : core.axes[i]) {
            if (label >= 0) {
                v.labels.push_back(label);
                v.cshape.push_back(l.shape[ax]);
                v.cstr.push_back(l.strides[ax]);
                p.sizes[static_cast<std::size_t>(label)] = l.shape[ax];
                ++ax;
            }
        }
        p.ops.push_back(std::move(v));
    }
    return p;
}

// The operand dense: [count][core axes...] (C order; `transpose`: the two
// core axes swapped). `once`: the batch axes left out (an operand that
// broadcasts along all of them).
Bytes gather(const Plan& p, const View& v, std::size_t es, bool transpose = false,
             bool once = false) {
    std::vector<std::int64_t> shape = once ? std::vector<std::int64_t>{} : p.batch;
    std::vector<std::int64_t> sstr = once ? std::vector<std::int64_t>{} : v.bstr;
    std::vector<std::int64_t> cs = v.cshape, cst = v.cstr;
    if (transpose && cs.size() == 2) {
        std::swap(cs[0], cs[1]);
        std::swap(cst[0], cst[1]);
    }
    shape.insert(shape.end(), cs.begin(), cs.end());
    sstr.insert(sstr.end(), cst.begin(), cst.end());
    std::int64_t total = 1;
    for (std::int64_t e : shape) {
        total *= e;
    }
    Bytes out(static_cast<std::size_t>(total) * es);
    const std::vector<std::int64_t> dstr = dense_strides(shape, es);
    copy_nd(shape.size(), shape.data(), v.base, sstr.data(), out.data(), dstr.data(), es);
    return out;
}

// Dense [count][core axes...] into the output `v` (`transpose` as gather).
void scatter(const Plan& p, const View& v, const std::byte* dense, std::size_t es,
             bool transpose = false) {
    std::vector<std::int64_t> shape = p.batch, dstr = v.bstr;
    std::vector<std::int64_t> cs = v.cshape, cst = v.cstr;
    if (transpose && cs.size() == 2) {
        std::swap(cs[0], cs[1]);
        std::swap(cst[0], cst[1]);
    }
    shape.insert(shape.end(), cs.begin(), cs.end());
    dstr.insert(dstr.end(), cst.begin(), cst.end());
    const std::vector<std::int64_t> sstr = dense_strides(shape, es);
    copy_nd(shape.size(), shape.data(), dense, sstr.data(), v.base, dstr.data(), es);
}

bool all_zero(const std::vector<std::int64_t>& v) {
    return std::all_of(v.begin(), v.end(), [](std::int64_t x) { return x == 0; });
}

bool is_complex(DType d) { return d == DType::Complex64 || d == DType::Complex128; }
bool is_double(DType d) { return d == DType::Float64 || d == DType::Complex128; }

void conj_in_place(Bytes& b, DType d) {
    if (d == DType::Complex128) {
        auto* x = reinterpret_cast<double*>(b.data());
        for (std::size_t i = 1; i < b.size() / sizeof(double); i += 2) {
            x[i] = -x[i];
        }
    } else if (d == DType::Complex64) {
        auto* x = reinterpret_cast<float*>(b.data());
        for (std::size_t i = 1; i < b.size() / sizeof(float); i += 2) {
            x[i] = -x[i];
        }
    }
}

std::int64_t log2_ceil(std::int64_t n) {
    std::int64_t l = 0;
    while ((std::int64_t{1} << l) < n) {
        ++l;
    }
    return std::max<std::int64_t>(l, 1);
}

// --- CLBlast calls by dtype ------------------------------------------------

void blas_check(int status, const char* what) {
    if (status != 0) {
        throw Error(std::string("CLBlast ") + what + ": status " + std::to_string(status));
    }
}

// C (rows-major, offsets in elements) = alpha A B + beta C, alpha/beta real
// numbers (complex types: alpha + 0i).
void gemm(const Clblast& b, DType d, cl_command_queue q, std::size_t m, std::size_t n,
          std::size_t k, double alpha, cl_mem A, std::size_t ao, std::size_t lda, cl_mem B,
          std::size_t bo, std::size_t ldb, double beta, cl_mem C, std::size_t co, std::size_t ldc) {
    int s = 0;
    switch (d) {
    case DType::Float32:
        s = b.sgemm(kRowMajor, kNoTrans, kNoTrans, m, n, k, static_cast<float>(alpha), A, ao, lda,
                    B, bo, ldb, static_cast<float>(beta), C, co, ldc, &q, nullptr);
        break;
    case DType::Float64:
        s = b.dgemm(kRowMajor, kNoTrans, kNoTrans, m, n, k, alpha, A, ao, lda, B, bo, ldb, beta, C,
                    co, ldc, &q, nullptr);
        break;
    case DType::Complex64: {
        cl_float2 a, be;
        a.s[0] = static_cast<float>(alpha), a.s[1] = 0;
        be.s[0] = static_cast<float>(beta), be.s[1] = 0;
        s = b.cgemm(kRowMajor, kNoTrans, kNoTrans, m, n, k, a, A, ao, lda, B, bo, ldb, be, C, co,
                    ldc, &q, nullptr);
        break;
    }
    case DType::Complex128: {
        cl_double2 a, be;
        a.s[0] = alpha, a.s[1] = 0;
        be.s[0] = beta, be.s[1] = 0;
        s = b.zgemm(kRowMajor, kNoTrans, kNoTrans, m, n, k, a, A, ao, lda, B, bo, ldb, be, C, co,
                    ldc, &q, nullptr);
        break;
    }
    default:
        throw Error("gemm: dtype");
    }
    blas_check(s, "gemm");
}

void gemm_batched(const Clblast& b, DType d, cl_command_queue q, std::size_t m, std::size_t n,
                  std::size_t k, cl_mem A, std::size_t as, cl_mem B, std::size_t bs, cl_mem C,
                  std::size_t cs, std::size_t count) {
    int s = 0;
    switch (d) {
    case DType::Float32:
        s = b.sgemm_batched(kRowMajor, kNoTrans, kNoTrans, m, n, k, 1.0f, A, 0, k, as, B, 0, n, bs,
                            0.0f, C, 0, n, cs, count, &q, nullptr);
        break;
    case DType::Float64:
        s = b.dgemm_batched(kRowMajor, kNoTrans, kNoTrans, m, n, k, 1.0, A, 0, k, as, B, 0, n, bs,
                            0.0, C, 0, n, cs, count, &q, nullptr);
        break;
    case DType::Complex64: {
        cl_float2 one, zero;
        one.s[0] = 1, one.s[1] = 0, zero.s[0] = 0, zero.s[1] = 0;
        s = b.cgemm_batched(kRowMajor, kNoTrans, kNoTrans, m, n, k, one, A, 0, k, as, B, 0, n, bs,
                            zero, C, 0, n, cs, count, &q, nullptr);
        break;
    }
    case DType::Complex128: {
        cl_double2 one, zero;
        one.s[0] = 1, one.s[1] = 0, zero.s[0] = 0, zero.s[1] = 0;
        s = b.zgemm_batched(kRowMajor, kNoTrans, kNoTrans, m, n, k, one, A, 0, k, as, B, 0, n, bs,
                            zero, C, 0, n, cs, count, &q, nullptr);
        break;
    }
    default:
        throw Error("gemm: dtype");
    }
    blas_check(s, "gemm (batched)");
}

// B = A^-1 B for a triangular A (left side), alpha 1.
void trsm(const Clblast& b, DType d, cl_command_queue q, int uplo, int diag, std::size_t m,
          std::size_t n, cl_mem A, std::size_t ao, std::size_t lda, cl_mem B, std::size_t bo,
          std::size_t ldb) {
    int s = 0;
    switch (d) {
    case DType::Float32:
        s = b.strsm(kRowMajor, kLeft, uplo, kNoTrans, diag, m, n, 1.0f, A, ao, lda, B, bo, ldb, &q,
                    nullptr);
        break;
    case DType::Float64:
        s = b.dtrsm(kRowMajor, kLeft, uplo, kNoTrans, diag, m, n, 1.0, A, ao, lda, B, bo, ldb, &q,
                    nullptr);
        break;
    case DType::Complex64: {
        cl_float2 one;
        one.s[0] = 1, one.s[1] = 0;
        s = b.ctrsm(kRowMajor, kLeft, uplo, kNoTrans, diag, m, n, one, A, ao, lda, B, bo, ldb, &q,
                    nullptr);
        break;
    }
    case DType::Complex128: {
        cl_double2 one;
        one.s[0] = 1, one.s[1] = 0;
        s = b.ztrsm(kRowMajor, kLeft, uplo, kNoTrans, diag, m, n, one, A, ao, lda, B, bo, ldb, &q,
                    nullptr);
        break;
    }
    default:
        throw Error("trsm: dtype");
    }
    blas_check(s, "trsm");
}

// Lower triangle of C -= A A^H (A: n x k).
void rank_k_down(const Clblast& b, DType d, cl_command_queue q, std::size_t n, std::size_t k,
                 cl_mem A, std::size_t ao, std::size_t lda, cl_mem C, std::size_t co,
                 std::size_t ldc) {
    int s = 0;
    switch (d) {
    case DType::Float32:
        s = b.ssyrk(kRowMajor, kLower, kNoTrans, n, k, -1.0f, A, ao, lda, 1.0f, C, co, ldc, &q,
                    nullptr);
        break;
    case DType::Float64:
        s = b.dsyrk(kRowMajor, kLower, kNoTrans, n, k, -1.0, A, ao, lda, 1.0, C, co, ldc, &q,
                    nullptr);
        break;
    case DType::Complex64:
        s = b.cherk(kRowMajor, kLower, kNoTrans, n, k, -1.0f, A, ao, lda, 1.0f, C, co, ldc, &q,
                    nullptr);
        break;
    case DType::Complex128:
        s = b.zherk(kRowMajor, kLower, kNoTrans, n, k, -1.0, A, ao, lda, 1.0, C, co, ldc, &q,
                    nullptr);
        break;
    default:
        throw Error("syrk: dtype");
    }
    blas_check(s, "syrk/herk");
}

// --- factorizations ---------------------------------------------------------

std::size_t pow2_at_least(std::size_t n) {
    std::size_t p = 1;
    while (p < n) {
        p *= 2;
    }
    return p;
}

std::size_t real_size(DType d) { return is_double(d) ? 8 : 4; }

// LU with partial pivoting of `count` n x n matrices in A (LAPACK's getrf).
// piv: count x n ints (0-based), info: count ints, both zeroed by the caller.
void getrf(Context& ctx, cl_command_queue q, DType d, const Clblast* blas, Buffer& A, int n,
           std::size_t count, Buffer& piv, Buffer& info) {
    cl_program prog = ctx.program(d);
    const std::uint64_t mstride = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
    if (n <= kSmallN || blas == nullptr) {
        KernelCall k(prog, "getf2");
        const std::size_t wg = ctx.work_group(
            k.get(), std::min<std::size_t>(256, pow2_at_least(static_cast<std::size_t>(n))));
        const int k0 = 0, mat0 = 0;
        k.arg(A.get())
            .arg(n)
            .arg(n)
            .arg(mstride)
            .arg(k0)
            .arg(n)
            .arg(piv.get())
            .arg(info.get())
            .arg(mat0)
            .local(wg * real_size(d))
            .local(wg * sizeof(int));
        k.run(q, count, wg);
        return;
    }
    std::size_t wg = 1, swg = 1;
    {
        KernelCall f(prog, "getf2");
        wg = ctx.work_group(f.get(), 256);
        KernelCall sw(prog, "laswp");
        swg = ctx.work_group(sw.get(), 64);
    }
    const std::size_t nn = static_cast<std::size_t>(n);
    for (std::size_t m = 0; m < count; ++m) {
        const int mat0 = static_cast<int>(m);
        const std::size_t off = m * nn * nn;
        for (int k0 = 0; k0 < n; k0 += kPanel) {
            const int b = std::min(kPanel, n - k0);
            KernelCall f(prog, "getf2");
            f.arg(A.get())
                .arg(n)
                .arg(n)
                .arg(mstride)
                .arg(k0)
                .arg(b)
                .arg(piv.get())
                .arg(info.get())
                .arg(mat0)
                .local(wg * real_size(d))
                .local(wg * sizeof(int));
            f.run(q, 1, wg);
            const int c0 = 0, c1 = n, s0 = k0, s1 = k0 + b, kb = k0, ke = k0 + b;
            const std::uint64_t poff = m * nn;
            const std::uint64_t aoff = off;
            KernelCall s(prog, "laswp");
            s.arg(A.get())
                .arg(n)
                .arg(aoff)
                .arg(c0)
                .arg(c1)
                .arg(s0)
                .arg(s1)
                .arg(piv.get())
                .arg(poff)
                .arg(kb)
                .arg(ke);
            s.run(q, (nn + swg - 1) / swg, swg);
            const std::size_t rest = nn - static_cast<std::size_t>(k0 + b);
            if (rest > 0) {
                const std::size_t kk = static_cast<std::size_t>(k0),
                                  bb = static_cast<std::size_t>(b);
                // A12 = L11^-1 A12
                trsm(*blas, d, q, kLower, kUnit, bb, rest, A.get(), off + kk * nn + kk, nn, A.get(),
                     off + kk * nn + kk + bb, nn);
                // A22 -= A21 A12
                gemm(*blas, d, q, rest, rest, bb, -1.0, A.get(), off + (kk + bb) * nn + kk, nn,
                     A.get(), off + kk * nn + kk + bb, nn, 1.0, A.get(),
                     off + (kk + bb) * nn + kk + bb, nn);
            }
        }
    }
}

// Solves with the LU factors (LAPACK's getrs): B (count x n x nrhs) in place.
void getrs(Context& ctx, cl_command_queue q, DType d, const Clblast* blas, Buffer& LU, int n,
           std::size_t count, Buffer& piv, Buffer& B, int nrhs) {
    cl_program prog = ctx.program(d);
    if (n <= kSmallN || blas == nullptr) {
        KernelCall k(prog, "getrs");
        const std::size_t wg = ctx.work_group(
            k.get(), std::min<std::size_t>(256, pow2_at_least(static_cast<std::size_t>(nrhs))));
        const int mat0 = 0;
        k.arg(LU.get()).arg(piv.get()).arg(B.get()).arg(n).arg(nrhs).arg(mat0);
        k.run(q, count, wg);
        return;
    }
    const std::size_t nn = static_cast<std::size_t>(n), nr = static_cast<std::size_t>(nrhs);
    KernelCall probe(prog, "laswp");
    const std::size_t swg = ctx.work_group(probe.get(), 64);
    for (std::size_t m = 0; m < count; ++m) {
        KernelCall s(prog, "laswp");
        const std::uint64_t boff = m * nn * nr, poff = m * nn;
        const int c0 = 0, c1 = nrhs, s0 = 0, s1 = 0, kb = 0, ke = n;
        s.arg(B.get())
            .arg(nrhs)
            .arg(boff)
            .arg(c0)
            .arg(c1)
            .arg(s0)
            .arg(s1)
            .arg(piv.get())
            .arg(poff)
            .arg(kb)
            .arg(ke);
        s.run(q, (nr + swg - 1) / swg, swg);
        trsm(*blas, d, q, kLower, kUnit, nn, nr, LU.get(), m * nn * nn, nn, B.get(), boff, nr);
        trsm(*blas, d, q, kUpper, kNonUnit, nn, nr, LU.get(), m * nn * nn, nn, B.get(), boff, nr);
    }
}

// Cholesky (lower) of `count` n x n matrices in A (LAPACK's potrf 'L' on the
// row-major lower triangle). info: count ints, zeroed by the caller.
void potrf(Context& ctx, cl_command_queue q, DType d, const Clblast* blas, Buffer& A, int n,
           std::size_t count, Buffer& info) {
    cl_program prog = ctx.program(d);
    const std::uint64_t mstride = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
    if (n <= kSmallN || blas == nullptr) {
        KernelCall k(prog, "potf2");
        const std::size_t wg = ctx.work_group(
            k.get(), std::min<std::size_t>(256, pow2_at_least(static_cast<std::size_t>(n))));
        const int k0 = 0, mat0 = 0;
        k.arg(A.get()).arg(n).arg(n).arg(mstride).arg(k0).arg(n).arg(info.get()).arg(mat0);
        k.run(q, count, wg);
        return;
    }
    const std::size_t nn = static_cast<std::size_t>(n);
    for (std::size_t m = 0; m < count; ++m) {
        const std::size_t off = m * nn * nn;
        for (int k0 = 0; k0 < n; k0 += kPanel) {
            const int b = std::min(kPanel, n - k0), mat0 = static_cast<int>(m);
            KernelCall k(prog, "potf2");
            const std::size_t wg = ctx.work_group(k.get(), 256);
            k.arg(A.get()).arg(n).arg(n).arg(mstride).arg(k0).arg(b).arg(info.get()).arg(mat0);
            k.run(q, 1, wg);
            const std::size_t rest = nn - static_cast<std::size_t>(k0 + b);
            if (rest > 0) {
                const std::size_t kk = static_cast<std::size_t>(k0),
                                  bb = static_cast<std::size_t>(b);
                rank_k_down(*blas, d, q, rest, bb, A.get(), off + (kk + bb) * nn + kk, nn, A.get(),
                            off + (kk + bb) * nn + kk + bb, nn);
            }
        }
    }
}

// --- NumPy's slogdet/det from the factored diagonal (umath_linalg.cpp) ------

template <class R> void slogdet_real(const R* diag, int n, const int* piv, R* sign, R* logdet) {
    int changes = 0;
    for (int i = 0; i < n; ++i) {
        changes += piv[i] != i;
    }
    R acc_sign = (changes % 2) ? R(-1) : R(1);
    R acc_logdet = 0;
    for (int i = 0; i < n; ++i) {
        R e = diag[i];
        if (e < 0) {
            acc_sign = -acc_sign;
            e = -e;
        }
        acc_logdet += std::log(e);
    }
    *sign = acc_sign;
    *logdet = acc_logdet;
}

template <class R>
void slogdet_complex(const std::complex<R>* diag, int n, const int* piv, std::complex<R>* sign,
                     R* logdet) {
    int changes = 0;
    for (int i = 0; i < n; ++i) {
        changes += piv[i] != i;
    }
    R sr = (changes % 2) ? R(-1) : R(1), si = 0;
    R acc_logdet = 0;
    for (int i = 0; i < n; ++i) {
        const R a = std::hypot(diag[i].real(), diag[i].imag()); // npy_cabs
        const R er = diag[i].real() / a, ei = diag[i].imag() / a;
        const R nr = sr * er - si * ei, ni = sr * ei + si * er;
        sr = nr, si = ni;
        acc_logdet += std::log(a);
    }
    *sign = {sr, si};
    *logdet = acc_logdet;
}

// --- the kernel -------------------------------------------------------------

class DeviceKernel : public Kernel {
  public:
    DeviceKernel(std::shared_ptr<Context> ctx, Op op, std::shared_ptr<Kernel> cpu,
                 numpy::GufuncCore core, std::vector<DType> dtypes, std::string invalid,
                 bool matmul_ranks)
        : ctx_(std::move(ctx)), op_(op), cpu_(std::move(cpu)), core_(std::move(core)),
          dtypes_(std::move(dtypes)), invalid_(std::move(invalid)), matmul_ranks_(matmul_ranks) {}

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        numpy::GufuncCore core = core_;
        std::size_t nin = 2;
        if (!matmul_ranks_) {
            nin = core.axes.size() - outputs.size();
        } else {
            const bool am = inputs[0].layout().shape.size() >= 2;
            const bool bp = inputs[1].layout().shape.size() >= 2;
            core.nlabels = 3;
            core.axes = {
                {am ? 0 : numpy::GufuncCore::missing(0), 1},
                {1, bp ? 2 : numpy::GufuncCore::missing(2)},
                {am ? 0 : numpy::GufuncCore::missing(0), bp ? 2 : numpy::GufuncCore::missing(2)}};
        }
        Plan p = make_plan(core, nin, inputs, outputs);
        bool empty = p.count == 0;
        for (std::int64_t s : p.sizes) {
            empty = empty || s == 0;
        }
        if (empty || work(p) < min_work()) {
            return cpu_->run(inputs, outputs);
        }
        // The task reads the FP flags after the kernel (NumPy's errstate):
        // only the host math NumPy's loops do too may leave flags (det's
        // exp and log, fft's normalization); OpenCL, CLBlast and VkFFT's
        // host code (plans) must not, so each op clears them before its
        // host math (det, fft) or at its end.
        try {
            KernelResult r = run_device(p);
            device_ops.fetch_add(1, std::memory_order_relaxed);
            return r;
        } catch (const Error&) {
        } catch (const std::bad_alloc&) {
        }
        device_fallbacks.fetch_add(1, std::memory_order_relaxed);
        std::feclearexcept(FE_ALL_EXCEPT);
        return cpu_->run(inputs, outputs);
    }

    bool in_place_ok() const override { return cpu_->in_place_ok(); }
    bool writes_all() const override { return cpu_->writes_all(); }
    const char* name() const override { return cpu_->name(); }
    const char* fp_name() const override { return cpu_->fp_name(); }
    DirectCost* direct_cost() override { return cpu_->direct_cost(); }
    KernelResult size(std::span<const Layout* const> in, std::span<Layout> out) override {
        return cpu_->size(in, out);
    }
    bool sizes_from_data() const override { return cpu_->sizes_from_data(); }
    KernelResult size_from_data(std::span<const Operand> in, std::span<Layout> out) override {
        return cpu_->size_from_data(in, out);
    }
    void layout(std::span<const Layout* const> in, std::span<Layout> out) override {
        cpu_->layout(in, out);
    }

  private:
    std::int64_t work(const Plan& p) const {
        const std::int64_t b = p.count;
        switch (op_) {
        case Op::Matmul:
            return b * p.sizes[0] * p.sizes[1] * p.sizes[2];
        case Op::Det:
        case Op::Slogdet:
        case Op::Inv:
        case Op::CholeskyLo:
        case Op::CholeskyUp:
        case Op::Solve1: {
            const std::int64_t n = p.sizes[0];
            return b * n * n * n;
        }
        case Op::Solve: {
            const std::int64_t n = p.sizes[0];
            return b * n * n * (n + p.sizes[1]);
        }
        default: {
            const std::int64_t n = std::max(p.sizes[static_cast<std::size_t>(core_.axes[0][0])],
                                            p.sizes[static_cast<std::size_t>(core_.axes[2][0])]);
            return b * n * log2_ceil(n) * 4;
        }
        }
    }

    KernelResult run_device(const Plan& p) {
        KernelResult r;
        switch (op_) {
        case Op::Matmul:
            matmul(p);
            break;
        case Op::Det:
        case Op::Slogdet:
            det(p); // clears the flags itself
            return {};
        case Op::Inv:
        case Op::Solve:
        case Op::Solve1:
            r = solve(p);
            break;
        case Op::CholeskyLo:
        case Op::CholeskyUp:
            r = cholesky(p);
            break;
        default:
            fft(p); // clears the flags itself
            return {};
        }
        std::feclearexcept(FE_ALL_EXCEPT);
        return r;
    }

    void check_fits(std::size_t bytes) const {
        if (bytes > ctx_->max_allocation()) {
            throw Error("buffer larger than the device's largest allocation");
        }
    }

    void matmul(const Plan& p) {
        const Clblast* blas = load_clblast();
        if (blas == nullptr) {
            throw Error("no CLBlast");
        }
        const DType d = dtypes_[0];
        const std::size_t es = itemsize(d);
        const std::size_t m = static_cast<std::size_t>(p.sizes[0]);
        const std::size_t k = static_cast<std::size_t>(p.sizes[1]);
        const std::size_t n = static_cast<std::size_t>(p.sizes[2]);
        const std::size_t count = static_cast<std::size_t>(p.count);
        const bool a1 = all_zero(p.ops[0].bstr), b1 = all_zero(p.ops[1].bstr);
        Bytes ha = gather(p, p.ops[0], es, false, a1);
        Bytes hb = gather(p, p.ops[1], es, false, b1);
        const std::size_t cbytes = count * m * n * es;
        check_fits(std::max({ha.size(), hb.size(), cbytes}));
        cl_command_queue q = ctx_->queue();
        Buffer A(*ctx_, ha.size()), B(*ctx_, hb.size()), C(*ctx_, cbytes);
        write(q, A, ha.data(), ha.size());
        write(q, B, hb.data(), hb.size());
        if (count == 1) {
            gemm(*blas, d, q, m, n, k, 1.0, A.get(), 0, k, B.get(), 0, n, 0.0, C.get(), 0, n);
        } else {
            gemm_batched(*blas, d, q, m, n, k, A.get(), a1 ? 0 : m * k, B.get(), b1 ? 0 : k * n,
                         C.get(), m * n, count);
        }
        Bytes hc(cbytes);
        read(q, C, hc.data(), cbytes);
        scatter(p, p.ops[2], hc.data(), es);
    }

    // The LU factors of input 0 on the device; returns n.
    int factor(const Plan& p, cl_command_queue q, Buffer& A, Buffer& piv, Buffer& info) {
        const DType d = dtypes_[0];
        const std::size_t es = itemsize(d);
        const int n = static_cast<int>(p.sizes[0]);
        const std::size_t count = static_cast<std::size_t>(p.count);
        Bytes ha = gather(p, p.ops[0], es);
        check_fits(ha.size());
        A = Buffer(*ctx_, ha.size());
        write(q, A, ha.data(), ha.size());
        std::vector<int> zeros(count * static_cast<std::size_t>(n), 0);
        piv = Buffer(*ctx_, zeros.size() * sizeof(int));
        write(q, piv, zeros.data(), zeros.size() * sizeof(int));
        info = Buffer(*ctx_, count * sizeof(int));
        write(q, info, zeros.data(), count * sizeof(int));
        const Clblast* blas = load_clblast();
        if (blas == nullptr && n > 4 * kSmallN) {
            throw Error("no CLBlast for a large matrix");
        }
        getrf(*ctx_, q, d, blas, A, n, count, piv, info);
        return n;
    }

    void det(const Plan& p) {
        cl_command_queue q = ctx_->queue();
        Buffer A, piv, info;
        const int n = factor(p, q, A, piv, info);
        const DType d = dtypes_[0];
        const std::size_t es = itemsize(d);
        const std::size_t count = static_cast<std::size_t>(p.count);
        const std::size_t nn = static_cast<std::size_t>(n);
        // the diagonals: one read per matrix row would cost a call each;
        // read the factors whole
        Bytes lu(count * nn * nn * es);
        read(q, A, lu.data(), lu.size());
        std::vector<int> hp(count * nn), hi(count);
        read(q, piv, hp.data(), hp.size() * sizeof(int));
        read(q, info, hi.data(), hi.size() * sizeof(int));
        const bool cplx = is_complex(d);
        const std::size_t rs = real_size(d);
        std::feclearexcept(FE_ALL_EXCEPT); // what follows is NumPy's host math
        Bytes out0(count * es), out1(count * rs);
        Bytes diag(nn * es);
        for (std::size_t m = 0; m < count; ++m) {
            for (std::size_t i = 0; i < nn; ++i) {
                std::memcpy(diag.data() + i * es, lu.data() + (m * nn * nn + i * nn + i) * es, es);
            }
            const int* pv = hp.data() + m * nn;
            std::byte* s = out0.data() + m * es;
            std::byte* l = out1.data() + m * rs;
            if (d == DType::Float64) {
                double sign = 0, logdet = -INFINITY;
                if (hi[m] == 0) {
                    slogdet_real(reinterpret_cast<const double*>(diag.data()), n, pv, &sign,
                                 &logdet);
                }
                const double v = op_ == Op::Det ? sign * std::exp(logdet) : sign;
                std::memcpy(s, &v, 8);
                std::memcpy(l, &logdet, 8);
            } else if (d == DType::Complex128) {
                std::complex<double> sign = 0;
                double logdet = -INFINITY;
                if (hi[m] == 0) {
                    slogdet_complex(reinterpret_cast<const std::complex<double>*>(diag.data()), n,
                                    pv, &sign, &logdet);
                }
                std::complex<double> v = sign;
                if (op_ == Op::Det) {
                    const double e = std::exp(logdet);
                    v = {sign.real() * e - sign.imag() * 0.0, sign.real() * 0.0 + sign.imag() * e};
                }
                std::memcpy(s, &v, 16);
                std::memcpy(l, &logdet, 8);
            } else if (d == DType::Float32) {
                float sign = 0, logdet = -INFINITY;
                if (hi[m] == 0) {
                    slogdet_real(reinterpret_cast<const float*>(diag.data()), n, pv, &sign,
                                 &logdet);
                }
                const float v = op_ == Op::Det ? sign * std::exp(logdet) : sign;
                std::memcpy(s, &v, 4);
                std::memcpy(l, &logdet, 4);
            } else {
                std::complex<float> sign = 0;
                float logdet = -INFINITY;
                if (hi[m] == 0) {
                    slogdet_complex(reinterpret_cast<const std::complex<float>*>(diag.data()), n,
                                    pv, &sign, &logdet);
                }
                std::complex<float> v = sign;
                if (op_ == Op::Det) {
                    const float e = std::exp(logdet);
                    v = {sign.real() * e - sign.imag() * 0.0f,
                         sign.real() * 0.0f + sign.imag() * e};
                }
                std::memcpy(s, &v, 8);
                std::memcpy(l, &logdet, 4);
            }
            (void)cplx;
        }
        scatter(p, p.ops[1], out0.data(), es);
        if (op_ == Op::Slogdet) {
            scatter(p, p.ops[2], out1.data(), rs);
        }
    }

    KernelResult solve(const Plan& p) {
        cl_command_queue q = ctx_->queue();
        Buffer A, piv, info;
        const int n = factor(p, q, A, piv, info);
        const DType d = dtypes_[0];
        const std::size_t es = itemsize(d);
        const std::size_t count = static_cast<std::size_t>(p.count);
        const std::size_t nn = static_cast<std::size_t>(n);
        std::vector<int> hi(count);
        read(q, info, hi.data(), hi.size() * sizeof(int));
        for (int v : hi) {
            if (v != 0) {
                return KernelResult{1, invalid_, 0};
            }
        }
        int nrhs = 1;
        Bytes hb;
        const View* out = nullptr;
        if (op_ == Op::Inv) {
            nrhs = n;
            hb.assign(count * nn * nn * es, std::byte{0});
            for (std::size_t m = 0; m < count; ++m) {
                for (std::size_t i = 0; i < nn; ++i) {
                    std::byte* e = hb.data() + (m * nn * nn + i * nn + i) * es;
                    if (d == DType::Float64 || d == DType::Complex128) {
                        const double one = 1.0;
                        std::memcpy(e, &one, 8);
                    } else {
                        const float one = 1.0f;
                        std::memcpy(e, &one, 4);
                    }
                }
            }
            out = &p.ops[1];
        } else {
            nrhs = op_ == Op::Solve ? static_cast<int>(p.sizes[1]) : 1;
            hb = gather(p, p.ops[1], es);
            out = &p.ops[2];
        }
        Buffer B(*ctx_, hb.size());
        write(q, B, hb.data(), hb.size());
        const Clblast* blas = load_clblast();
        getrs(*ctx_, q, d, blas, A, n, count, piv, B, nrhs);
        read(q, B, hb.data(), hb.size());
        scatter(p, *out, hb.data(), es);
        return {};
    }

    KernelResult cholesky(const Plan& p) {
        const DType d = dtypes_[0];
        const std::size_t es = itemsize(d);
        const bool up = op_ == Op::CholeskyUp;
        const int n = static_cast<int>(p.sizes[0]);
        const std::size_t nn = static_cast<std::size_t>(n);
        const std::size_t count = static_cast<std::size_t>(p.count);
        // upper: the lower factor of A^H (whose lower triangle is the
        // conjugated upper triangle of A), U = L^H
        Bytes ha = gather(p, p.ops[0], es, up);
        if (up) {
            conj_in_place(ha, d);
        }
        check_fits(ha.size());
        cl_command_queue q = ctx_->queue();
        Buffer A(*ctx_, ha.size());
        write(q, A, ha.data(), ha.size());
        std::vector<int> hi(count, 0);
        Buffer info(*ctx_, count * sizeof(int));
        write(q, info, hi.data(), count * sizeof(int));
        const Clblast* blas = load_clblast();
        if (blas == nullptr && n > 4 * kSmallN) {
            throw Error("no CLBlast for a large matrix");
        }
        potrf(*ctx_, q, d, blas, A, n, count, info);
        read(q, info, hi.data(), count * sizeof(int));
        for (int v : hi) {
            if (v != 0) {
                return KernelResult{1, invalid_, 0};
            }
        }
        read(q, A, ha.data(), ha.size());
        for (std::size_t m = 0; m < count; ++m) {
            for (std::size_t i = 0; i < nn; ++i) {
                std::memset(ha.data() + (m * nn * nn + i * nn + i + 1) * es, 0, (nn - i - 1) * es);
            }
        }
        if (up) {
            conj_in_place(ha, d);
        }
        scatter(p, p.ops[1], ha.data(), es, up);
        return {};
    }

    void fft(const Plan& p) {
        const DType din = dtypes_[0], dout = dtypes_[2];
        const bool dbl = is_double(din);
        const std::size_t rs = dbl ? 8 : 4, cs = 2 * rs;
        const std::size_t count = static_cast<std::size_t>(p.count);
        const std::int64_t nin = p.sizes[static_cast<std::size_t>(core_.axes[0][0])];
        const std::int64_t nout = p.sizes[static_cast<std::size_t>(core_.axes[2][0])];
        std::int64_t npts = nout;
        FftKind kind = FftKind::Forward;
        switch (op_) {
        case Op::Ifft:
            kind = FftKind::Backward;
            break;
        case Op::RfftEven:
            kind = FftKind::RealForward;
            npts = 2 * (nout - 1);
            break;
        case Op::RfftOdd:
            kind = FftKind::RealForward;
            npts = 2 * nout - 1;
            break;
        case Op::Irfft:
            kind = FftKind::RealBackward;
            break;
        default:
            break;
        }
        if (npts < 2) {
            throw Error("fft: too few points for the device");
        }
        const bool real = kind == FftKind::RealForward || kind == FftKind::RealBackward;
        const std::int64_t h = npts / 2 + 1;
        // one row: npts complex values, or the in-place real layout of h
        // complex values
        const std::int64_t row_bytes = static_cast<std::int64_t>(real ? h * cs : npts * cs);
        Bytes host(count * static_cast<std::size_t>(row_bytes), std::byte{0});
        check_fits(host.size());
        const View& in = p.ops[0];
        {
            // the input rows into host: min(nin, length) elements each
            std::int64_t take = 0, es = 0;
            if (kind == FftKind::RealForward) {
                take = std::min(nin, npts), es = static_cast<std::int64_t>(rs);
            } else if (kind == FftKind::RealBackward) {
                take = std::min(nin, h), es = static_cast<std::int64_t>(cs);
            } else {
                take = std::min(nin, npts), es = static_cast<std::int64_t>(cs);
            }
            std::vector<std::int64_t> shape = p.batch, sstr = in.bstr;
            shape.push_back(take);
            sstr.push_back(in.cstr[0]);
            std::vector<std::int64_t> dstr = dense_strides(p.batch, 1);
            for (std::int64_t& s : dstr) {
                s *= row_bytes;
            }
            dstr.push_back(es);
            copy_nd(shape.size(), shape.data(), in.base, sstr.data(), host.data(), dstr.data(),
                    static_cast<std::size_t>(es));
            if (kind == FftKind::RealBackward) {
                // pocketfft's irfft ignores the imaginary parts of the zero
                // and (n even) the Nyquist frequency
                for (std::size_t r = 0; r < count; ++r) {
                    std::byte* row = host.data() + r * static_cast<std::size_t>(row_bytes);
                    std::memset(row + rs, 0, rs);
                    if (npts % 2 == 0 && npts / 2 < nin) {
                        std::memset(row + static_cast<std::size_t>(npts / 2) * cs + rs, 0, rs);
                    }
                }
            }
        }
        cl_command_queue q = ctx_->queue();
        Buffer buf(*ctx_, host.size());
        write(q, buf, host.data(), host.size());
        opencl::fft(*ctx_, q, kind, dbl, static_cast<std::uint64_t>(npts),
                    static_cast<std::uint64_t>(count), buf);
        read(q, buf, host.data(), host.size());
        std::feclearexcept(FE_ALL_EXCEPT); // what follows is pocketfft's scaling
        // NumPy's normalization factor, one per batch element (input 1)
        Bytes fb = gather(p, p.ops[1], rs);
        const std::int64_t len = kind == FftKind::RealBackward ? npts : nout;
        const std::size_t es = kind == FftKind::RealBackward ? rs : cs;
        for (std::size_t r = 0; r < count; ++r) {
            std::byte* row = host.data() + r * static_cast<std::size_t>(row_bytes);
            const std::size_t nreal = static_cast<std::size_t>(len) * es / rs;
            if (dbl) {
                double f;
                std::memcpy(&f, fb.data() + r * rs, 8);
                if (f != 1.0) {
                    auto* x = reinterpret_cast<double*>(row);
                    for (std::size_t i = 0; i < nreal; ++i) {
                        x[i] *= f;
                    }
                }
            } else {
                float f;
                std::memcpy(&f, fb.data() + r * rs, 4);
                if (f != 1.0f) {
                    auto* x = reinterpret_cast<float*>(row);
                    for (std::size_t i = 0; i < nreal; ++i) {
                        x[i] *= f;
                    }
                }
            }
        }
        (void)dout;
        const View& out = p.ops[2];
        std::vector<std::int64_t> shape = p.batch, dstr = out.bstr;
        shape.push_back(len);
        dstr.push_back(out.cstr[0]);
        std::vector<std::int64_t> sstr = dense_strides(p.batch, 1);
        for (std::int64_t& s : sstr) {
            s *= row_bytes;
        }
        sstr.push_back(static_cast<std::int64_t>(es));
        copy_nd(shape.size(), shape.data(), host.data(), sstr.data(), out.base, dstr.data(), es);
    }

    std::shared_ptr<Context> ctx_;
    Op op_;
    std::shared_ptr<Kernel> cpu_;
    numpy::GufuncCore core_;
    std::vector<DType> dtypes_;
    std::string invalid_;
    bool matmul_ranks_;
};

} // namespace

Op gufunc_op(const char* name, int nin, int nout) {
    struct Entry {
        const char* name;
        int nin, nout;
        Op op;
    };
    static const Entry table[] = {
        {"matmul", 2, 1, Op::Matmul},
        {"det", 1, 1, Op::Det},
        {"slogdet", 1, 2, Op::Slogdet},
        {"inv", 1, 1, Op::Inv},
        {"solve", 2, 1, Op::Solve},
        {"solve1", 2, 1, Op::Solve1},
        {"cholesky_lo", 1, 1, Op::CholeskyLo},
        {"cholesky_up", 1, 1, Op::CholeskyUp},
        {"fft", 2, 1, Op::Fft},
        {"ifft", 2, 1, Op::Ifft},
        {"rfft_n_even", 2, 1, Op::RfftEven},
        {"rfft_n_odd", 2, 1, Op::RfftOdd},
        {"irfft", 2, 1, Op::Irfft},
    };
    if (name == nullptr) {
        return Op::None;
    }
    for (const Entry& e : table) {
        if (std::strcmp(name, e.name) == 0 && nin == e.nin && nout == e.nout) {
            return e.op;
        }
    }
    return Op::None;
}

bool supports(const Context& ctx, Op op, DType dtype) {
    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Complex64 &&
        dtype != DType::Complex128) {
        return false;
    }
    if (is_double(dtype) && !ctx.fp64()) {
        return false;
    }
    switch (op) {
    case Op::None:
        return false;
    case Op::Matmul:
        return load_clblast() != nullptr;
    case Op::Fft:
    case Op::Ifft:
        return is_complex(dtype);
    case Op::RfftEven:
    case Op::RfftOdd:
        return !is_complex(dtype);
    case Op::Irfft:
        return is_complex(dtype);
    default:
        return true;
    }
}

std::shared_ptr<Kernel> make_gufunc_kernel(std::shared_ptr<Context> ctx, Op op,
                                           std::shared_ptr<Kernel> cpu,
                                           const numpy::GufuncCore& core, std::vector<DType> dtypes,
                                           std::string invalid_error) {
    return std::make_shared<DeviceKernel>(std::move(ctx), op, std::move(cpu), core,
                                          std::move(dtypes), std::move(invalid_error), false);
}

std::shared_ptr<Kernel> make_matmul_kernel(std::shared_ptr<Context> ctx,
                                           std::shared_ptr<Kernel> cpu, DType dtype) {
    return std::make_shared<DeviceKernel>(
        std::move(ctx), Op::Matmul, std::move(cpu), numpy::GufuncCore{},
        std::vector<DType>{dtype, dtype, dtype}, std::string(), true);
}

} // namespace viproc::opencl
