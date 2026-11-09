// viproc's own OpenCL kernels: LU with partial pivoting (getf2 panels,
// row interchanges, triangular solves of small systems) and Cholesky panels.
// Built per dtype with -DVP_F32/-DVP_F64/-DVP_C64/-DVP_C128 (Context::program).
// Matrices are row-major, `lda` elements per row, `mstride` elements apart;
// work-group g handles matrix mat0 + g. Large matrices use CLBlast for the
// trailing updates (cl_kernels.cpp); these kernels then factor one panel.
namespace viproc::opencl {

extern const char* const kLinalgSource;
const char* const kLinalgSource = R"CLC(
#if defined(VP_F64) || defined(VP_C128)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
typedef double2 real2;
#else
typedef float real;
typedef float2 real2;
#endif

#if defined(VP_C64) || defined(VP_C128)
typedef real2 T;
#define MUL(a, b) ((T)((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x))
#define CONJ(a) ((T)((a).x, -(a).y))
#define ABS1(a) (fabs((a).x) + fabs((a).y))
#define ISZERO(a) ((a).x == 0 && (a).y == 0)
#define RE(a) ((a).x)
#define FROM_REAL(r) ((T)((r), 0))
#define SCALE(a, r) ((a) * (r))
inline T DIV(T a, T b) {
    /* Smith's algorithm, as LAPACK's zladiv avoids overflow */
    if (fabs(b.y) <= fabs(b.x)) {
        real e = b.y / b.x, f = b.x + b.y * e;
        return (T)((a.x + a.y * e) / f, (a.y - a.x * e) / f);
    }
    real e = b.x / b.y, f = b.y + b.x * e;
    return (T)((a.x * e + a.y) / f, (a.y * e - a.x) / f);
}
#else
typedef real T;
#define MUL(a, b) ((a) * (b))
#define CONJ(a) (a)
#define ABS1(a) fabs(a)
#define ISZERO(a) ((a) == 0)
#define RE(a) (a)
#define FROM_REAL(r) (r)
#define SCALE(a, r) ((a) * (r))
#define DIV(a, b) ((a) / (b))
#endif

/* LU with partial pivoting of the panel of columns [k0, k0 + b) over rows
 * [k0, n) (LAPACK's getf2 on a panel): pivots as LAPACK's i?amax picks them
 * (largest |re| + |im|, the first of equal ones; NaN never wins over a
 * number), row interchanges within the panel's columns, the column scaled
 * by the pivot's reciprocal, the panel's trailing part updated. piv[k] is
 * the 0-based row swapped with row k; info is the first zero pivot (1-based,
 * 0: none). The factorization goes on after a zero pivot, as LAPACK's. */
__kernel void getf2(__global T* A, int n, int lda, ulong mstride, int k0, int b,
                    __global int* piv, __global int* info, int mat0,
                    __local real* lv, __local int* li) {
    const ulong mat = (ulong)mat0 + get_group_id(0);
    A += mat * mstride;
    piv += mat * (ulong)n;
    info += mat;
    const int lid = (int)get_local_id(0), wg = (int)get_local_size(0);
    for (int k = k0; k < k0 + b; ++k) {
        real best = -2;
        int bi = n;
        for (int i = k + lid; i < n; i += wg) {
            real v = ABS1(A[(ulong)i * lda + k]);
            if (!(v >= 0)) {
                v = -1; /* NaN */
            }
            if (v > best) {
                best = v;
                bi = i;
            }
        }
        lv[lid] = best;
        li[lid] = bi;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = wg / 2; s > 0; s /= 2) {
            if (lid < s) {
                real o = lv[lid + s];
                int oi = li[lid + s];
                if (o > lv[lid] || (o == lv[lid] && oi < li[lid])) {
                    lv[lid] = o;
                    li[lid] = oi;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const int p = li[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        const T pv = A[(ulong)p * lda + k];
        const bool nz = !ISZERO(pv);
        if (lid == 0) {
            piv[k] = p;
            if (!nz && *info == 0) {
                *info = k + 1;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE); /* every work-item read pv */
        /* no barrier under a condition: pocl wants them in straight code */
        if (nz && p != k) {
            for (int j = k0 + lid; j < k0 + b; j += wg) {
                T t = A[(ulong)k * lda + j];
                A[(ulong)k * lda + j] = A[(ulong)p * lda + j];
                A[(ulong)p * lda + j] = t;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (nz) {
            const T r = DIV(FROM_REAL(1), pv);
            for (int i = k + 1 + lid; i < n; i += wg) {
                A[(ulong)i * lda + k] = MUL(A[(ulong)i * lda + k], r);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        const int nc = k0 + b - k - 1, nr = n - k - 1;
        if (nc > 0) {
            const long total = (long)nr * nc;
            for (long t = lid; t < total; t += wg) {
                const int i = k + 1 + (int)(t / nc), j = k + 1 + (int)(t % nc);
                A[(ulong)i * lda + j] -= MUL(A[(ulong)i * lda + k], A[(ulong)k * lda + j]);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

/* The row interchanges piv[kb..ke) applied to the columns [c0, c1) of a
 * matrix outside [s0, s1) (LAPACK's laswp), one work-item per column. */
__kernel void laswp(__global T* A, int lda, ulong off, int c0, int c1, int s0, int s1,
                    __global const int* piv, ulong poff, int kb, int ke) {
    const int j = c0 + (int)get_global_id(0);
    if (j >= c1 || (j >= s0 && j < s1)) {
        return;
    }
    A += off;
    piv += poff;
    for (int k = kb; k < ke; ++k) {
        const int p = piv[k];
        if (p != k) {
            T t = A[(ulong)k * lda + j];
            A[(ulong)k * lda + j] = A[(ulong)p * lda + j];
            A[(ulong)p * lda + j] = t;
        }
    }
}

/* Solves A X = B with A's LU factors (LAPACK's getrs, no transpose): the
 * interchanges, then the unit lower and the upper triangular solve. One
 * work-group per matrix, a work-item per column of B (n x nrhs, row-major). */
__kernel void getrs(__global const T* LU, __global const int* piv, __global T* B, int n,
                    int nrhs, int mat0) {
    const ulong mat = (ulong)mat0 + get_group_id(0);
    LU += mat * (ulong)n * n;
    piv += mat * (ulong)n;
    B += mat * (ulong)n * nrhs;
    const int lid = (int)get_local_id(0), wg = (int)get_local_size(0);
    for (int c = lid; c < nrhs; c += wg) {
        for (int k = 0; k < n; ++k) {
            const int p = piv[k];
            if (p != k) {
                T t = B[(ulong)k * nrhs + c];
                B[(ulong)k * nrhs + c] = B[(ulong)p * nrhs + c];
                B[(ulong)p * nrhs + c] = t;
            }
        }
        for (int i = 1; i < n; ++i) {
            T s = B[(ulong)i * nrhs + c];
            for (int j = 0; j < i; ++j) {
                s -= MUL(LU[(ulong)i * n + j], B[(ulong)j * nrhs + c]);
            }
            B[(ulong)i * nrhs + c] = s;
        }
        for (int i = n - 1; i >= 0; --i) {
            T s = B[(ulong)i * nrhs + c];
            for (int j = i + 1; j < n; ++j) {
                s -= MUL(LU[(ulong)i * n + j], B[(ulong)j * nrhs + c]);
            }
            B[(ulong)i * nrhs + c] = DIV(s, LU[(ulong)i * n + i]);
        }
    }
}

/* Cholesky (lower, LAPACK's potf2) of the panel of columns [k0, k0 + b)
 * over rows [k0, n), reading the lower triangle only (the diagonal's real
 * part). info: the first column whose pivot is not positive (or NaN),
 * 1-based; the factorization stops there, as LAPACK's. */
__kernel void potf2(__global T* A, int n, int lda, ulong mstride, int k0, int b,
                    __global int* info, int mat0) {
    const ulong mat = (ulong)mat0 + get_group_id(0);
    A += mat * mstride;
    info += mat;
    const int lid = (int)get_local_id(0), wg = (int)get_local_size(0);
    /* stop: a pivot failed (here or in an earlier panel); the loop goes on
     * without work, so every barrier is reached in straight code */
    bool stop = *info != 0;
    for (int k = k0; k < k0 + b; ++k) {
        real akk = stop ? 1 : RE(A[(ulong)k * lda + k]);
        if (!(akk > 0)) {
            stop = true;
            if (lid == 0) {
                *info = k + 1;
            }
        }
        akk = sqrt(akk);
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (!stop) {
            if (lid == 0) {
                A[(ulong)k * lda + k] = FROM_REAL(akk);
            }
            const real r = 1 / akk;
            for (int i = k + 1 + lid; i < n; i += wg) {
                A[(ulong)i * lda + k] = SCALE(A[(ulong)i * lda + k], r);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        const int nc = k0 + b - k - 1, nr = n - k - 1;
        if (!stop && nc > 0) {
            const long total = (long)nr * nc;
            for (long t = lid; t < total; t += wg) {
                const int i = k + 1 + (int)(t / nc), j = k + 1 + (int)(t % nc);
                if (j <= i) {
                    A[(ulong)i * lda + j] -=
                        MUL(A[(ulong)i * lda + k], CONJ(A[(ulong)j * lda + k]));
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
/* --- device-resident arrays (cl_kernels.cpp: stage, deliver) -------------
 * Elements are copied as 32-bit words: every dtype on the device is a
 * multiple of 4 bytes, and offsets and strides are (checked on the host). */

/* An n-d block (nd <= 8; nd = 0: one element) of elements of `words` words
 * between two strided layouts, offsets and strides in bytes. One work-item
 * per element, `total` of them. */
__kernel void copy_nd(__global const uint* src, ulong soff, __global uint* dst, ulong doff,
                      int nd, long8 shape, long8 sstr, long8 dstr, int words, ulong total) {
    const ulong i = get_global_id(0);
    if (i < total) {
        long sh[8], ss[8], ds[8];
        vstore8(shape, 0, sh);
        vstore8(sstr, 0, ss);
        vstore8(dstr, 0, ds);
        long s = (long)soff, d = (long)doff;
        ulong r = i;
        for (int a = nd - 1; a >= 0; --a) {
            const ulong e = (ulong)sh[a];
            const long k = (long)(r % e);
            r /= e;
            s += k * ss[a];
            d += k * ds[a];
        }
        for (int w = 0; w < words; ++w) {
            dst[d / 4 + w] = src[s / 4 + w];
        }
    }
}

/* n words of zeros. */
__kernel void fill_zero(__global uint* a, ulong n) {
    const ulong i = get_global_id(0);
    if (i < n) {
        a[i] = 0;
    }
}

/* Conjugates n elements (real types: nothing to do). */
__kernel void conj_all(__global T* a, ulong n) {
    const ulong i = get_global_id(0);
    if (i < n) {
        a[i] = CONJ(a[i]);
    }
}

/* Zeroes the strict upper triangle of `count` row-major n x n matrices. */
__kernel void zero_upper(__global T* a, int n, ulong count) {
    const ulong i = get_global_id(0);
    const ulong nn = (ulong)n * (ulong)n;
    if (i < count * nn) {
        const ulong e = i % nn;
        if (e % (ulong)n > e / (ulong)n) {
            a[i] = FROM_REAL(0);
        }
    }
}

/* Row r (`stride` reals apart) of `count` rows: its first nreal reals times
 * f[r]. */
__kernel void scale_rows(__global real* x, ulong stride, ulong nreal, __global const real* f,
                         ulong count) {
    const ulong i = get_global_id(0);
    if (i < count * nreal) {
        const ulong r = i / nreal;
        x[r * stride + i % nreal] *= f[r];
    }
}

/* pocketfft's irfft ignores the imaginary parts of the zero and (npts even,
 * the input long enough) the Nyquist frequency: zeroes them in each row
 * (`stride` reals apart). */
__kernel void irfft_prep(__global real* x, ulong stride, int npts, long nin, ulong count) {
    const ulong r = get_global_id(0);
    if (r < count) {
        x[r * stride + 1] = 0;
        if (npts % 2 == 0 && npts / 2 < nin) {
            x[r * stride + (ulong)(npts / 2) * 2 + 1] = 0;
        }
    }
}
)CLC";

} // namespace viproc::opencl
