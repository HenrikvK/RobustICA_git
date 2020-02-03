#include "PCA.h"
#include <complex> // for complex numbers

// keep include/define in that order
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float> 
#define MKL_Complex16 std::complex<double> // calculate roots in higher accuracy
#include <mkl.h>

using namespace std;

void PCA(float* a, int lda, int m, int n) {
    lapack_int ldu = 1, ldvt = n;
    float* superb = (float*)malloc((m - 1) * sizeof(float));
    // Local arrays 
    float* s = (float*)malloc((n) * sizeof(float));
    float* u = (float*)malloc(1 * sizeof(float));
    float* vt = (float*)malloc(ldvt * sizeof(float));

    // Compute SVD 
    LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'N', 'O', m, n, a, lda,
        s, u, ldu, vt, ldvt, superb);

    free(superb);
    free(s);
    free(u);
    free(vt);
}
void PCA(complex<float>* a, int lda, int m, int n) {
    lapack_int ldu = 1, ldvt = n;
    float* superb = (float*)malloc((m - 1) * sizeof(float));
    // Local arrays 
    float* s = (float*)malloc((n) * sizeof(float));
    complex<float>* u = (complex<float>*)malloc(1 * sizeof(complex<float>));
    complex<float>* vt = (complex<float>*)malloc(n * sizeof(complex<float>));

    // Compute SVD 
    LAPACKE_cgesvd(LAPACK_ROW_MAJOR, 'N', 'O', m, n, a, lda,
        s, u, ldu, vt, ldvt, superb);

    free(superb);
    free(s);
    free(u);
    free(vt);
}
