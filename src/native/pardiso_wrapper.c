/* Windows GNU/Clang ABI guard: the tested MKL 2025.0 PARDISO runtime modified callee-saved
 * xmm9..xmm15 registers, corrupting values kept in those registers by the caller.
 * The empty asm with xmm6..15 clobbers forces this wrapper's prologue/epilogue to save/restore them.
 * Use: compile the solver with -DPARDISO=pardiso_abi_safe and link this wrapper. */
#include <mkl.h>
void pardiso_abi_safe(_MKL_DSS_HANDLE_t pt, const MKL_INT *maxfct, const MKL_INT *mnum, const MKL_INT *mtype,
                      const MKL_INT *phase, const MKL_INT *n, const void *a, const MKL_INT *ia, const MKL_INT *ja,
                      MKL_INT *perm, const MKL_INT *nrhs, MKL_INT *iparm, const MKL_INT *msglvl, void *b, void *x,
                      MKL_INT *error)
{
    pardiso(pt, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, iparm, msglvl, b, x, error);
#if defined(_WIN32) && (defined(__GNUC__) || defined(__clang__))
    __asm__ volatile("" ::: "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory");
#endif
}
