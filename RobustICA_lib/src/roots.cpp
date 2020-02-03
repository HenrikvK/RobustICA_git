#include "roots.h"
#include <complex> // for complex numbers

// keep include/define in that order
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float> 
#define MKL_Complex16 std::complex<double> // calculate roots in higher accuracy
#include <mkl.h>

using namespace std;

void roots(float* coeffs, int nr_coeffs, double* wr) {
    int matsz = nr_coeffs - 1;   //int matsz = coeffs.size() - 1;

    double* companion_mat = (double*)malloc(matsz * matsz * sizeof(double)); // MatrixXd companion_mat(matsz, matsz);
    for (int n = 0; n < matsz; n++) {
        for (int m = 0; m < matsz; m++) {
            if (n == m + 1) {
                companion_mat[n * matsz + m] = 1.0;
            }
            else {
                companion_mat[n * matsz + m] = 0;
            }
            if (m == matsz - 1)
                companion_mat[n * matsz + m] = (double)-coeffs[matsz - n] / (double)coeffs[0];
        }
    }

    double* wi = (double*)malloc(matsz * sizeof(double));   // imaginary parts of eigenvalue
    // solve eigenvalue problem:
    int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', matsz, companion_mat, matsz, wr, wi, nullptr, 5, nullptr, 5);
    std::free(wi);
    std::free(companion_mat);
}
void roots(complex<float>* coeffs, int nr_coeffs, complex<double>* r) {
    int matsz = nr_coeffs - 1;   //int matsz = coeffs.size() - 1;

    MKL_Complex16* companion_mat = (MKL_Complex16*)malloc(matsz * matsz * sizeof(MKL_Complex16)); // MatrixXd companion_mat(matsz, matsz);
    for (int n = 0; n < matsz; n++) {
        for (int m = 0; m < matsz; m++) {
            if (n == m + 1) {
                companion_mat[n * matsz + m] = MKL_Complex16(1.0, 0);
            }
            else {
                companion_mat[n * matsz + m] = MKL_Complex16(0, 0);
            }
            if (m == matsz - 1)
                companion_mat[n * matsz + m] = ((MKL_Complex16)-coeffs[matsz - n]) / ((MKL_Complex16)coeffs[0]);
        }
    }

    // solve eigenvalue problem:
    int info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'N', matsz, companion_mat, matsz, r, nullptr, 5, nullptr, 5);
    std::free(companion_mat);
}