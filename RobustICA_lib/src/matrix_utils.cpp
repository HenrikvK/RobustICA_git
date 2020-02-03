#include "matrix_utils.h"
#include <complex> // for complex numbers

// keep include/define in that order
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float> 
#define MKL_Complex16 std::complex<double> // calculate roots in higher accuracy
#include <mkl.h>

using namespace std;

void polyval(float* coeffs, int nr_coeffs, double* xvalues, int nr_values, double* yvalues) {
    for (int i = 0; i < nr_values; i++) {
        yvalues[i] = 0;
        for (int j = 0; j < nr_coeffs; j++) {
            yvalues[i] += ((double)coeffs[nr_coeffs - j - 1]) * pow(xvalues[i], j);
        }
    }
}

int find_nonzero(double* a, int len, int* non_zero) {
    double epsf = std::numeric_limits<double>::epsilon();
    int nr_non_zero = 0;
    // find the nonzero entries in a:
    for (int i = 0; i < len; i++) {
        if (a[i] > epsf) {
            non_zero[nr_non_zero] = i;
            nr_non_zero++;
        }
    }
    return nr_non_zero;
}

void values_at_indices(double* a, int* indices, int nr_indices) {
    for (int i = 0; i < nr_indices; i++) {
        a[i] = a[indices[i]];
    }
}

void init_identity(float* a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                a[i * n + j] = 1;
            }
            else {
                a[i * n + j] = 0;
            }
        }
    }
}
void init_identity(std::complex<float>* a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                a[i * n + j] = complex<float>(1, 0);
            }
            else {
                a[i * n + j] = complex<float>(0, 0);
            }
        }
    }
}

void init_zeros(float* a, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = 0;
        }
    }
}
void init_zeros(std::complex<float>* a, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = std::complex<float>(0, 0);
        }
    }
}

void copy_matrix(float* a, int m, int n, float* b) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = a[i * n + j];
        }
    }
}
void copy_matrix(std::complex<float>* a, int m, int n, std::complex<float>* b) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = a[i * n + j];
        }
    }
}

void subtract_mean(float* a, int lda, int n, int T) {
    float mean;
    for (int i = 0; i < n; i++) {
        mean = compute_mean(a + lda * i, T);
        for (int j = 0; j < T; j++) {
            a[i * lda + j] -= mean;
        }
    }
}
void subtract_mean(std::complex<float>* a, int lda, int n, int T) {
    std::complex<float> mean;
    for (int i = 0; i < n; i++) {
        mean = compute_mean(a + lda * i, T);
        for (int j = 0; j < T; j++) {
            a[i * lda + j] -= mean;
        }
    }
}

float compute_mean(float* a, int len) {
    float mean = 0;
    for (int i = 0; i < len; i++) {
        mean += a[i];
    }
    mean /= len;
    return mean;
}
std::complex<float> compute_mean(std::complex<float>* a, int len) {
    std::complex<float>  mean = 0;
    for (int i = 0; i < len; i++) {
        mean += a[i];
    }
    mean /= (float)len;
    return mean;
}

void normalize(float* a, int len) {
    float norm = compute_norm(a, len);
    for (int i = 0; i < len; i++) {
        a[i] = a[i] / norm;
    }
}
void normalize(std::complex<float>* a, int len) {
    float norm = compute_norm(a, len);
    for (int i = 0; i < len; i++) {
        a[i] = a[i] / norm;
    }
}

float compute_norm(float* a, int len) {
    float sum2 = 0;
    float norm;
    for (int i = 0; i < len; i++) {
        sum2 += a[i] * a[i];
    }
    norm = sqrt(sum2);
    return norm;
}
float compute_norm(complex<float>* a, int len) {
    float sum2 = 0;
    float norm;
    for (int i = 0; i < len; i++) {
        sum2 += abs(a[i]) * abs(a[i]);
    }
    norm = sqrt(sum2);
    return norm;
}

void print_matrix(const char* desc, int m, int n, float* a, int lda) {
    // print only maximum a 5x5 matrix:
    if (m > 7) {
        m = 5;
        n = 5;
    }
    int i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) printf(" %6.8f", a[i * lda + j]);
        printf("\n");
    }
}
void print_matrix(const char* desc, int m, int n, double* a, int lda) {
    // print max a 5x5 matrix:
    if (m > 7) {
        m = 5;
        n = 5;
    }
    int i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) printf(" %6.8f", a[i * lda + j]);
        printf("\n");
    }
}
void print_matrix(const char* desc, int m, int n, std::complex<float>* a, int lda) {
    // print max a 5x5 matrix:
    if (m > 7) {
        m = 5;
        n = 5;
    }
    int i, j;
    printf("\n %s\n", desc);
    std::complex<float> temp;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            temp = a[i * lda + j];
            printf(" (%6.8f,%6.8f)", std::real(temp), std::imag(temp));
        }
        printf("\n");
    }
}
