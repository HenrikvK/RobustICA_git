#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_

#include <complex> // for complex numbers

using namespace std;

// evaluate a polynomial at specific values
// input:
//          coeffs:         coefficients of the polynomial (y = coeffs[0]*x^(nr_coeffs-1)+...) (length: nr_coeffs)
//          nr_coeffs:      number of coefficients (= degree of polynomial + 1) 
//          xvalues:        points at which the polynomial is evaluated (double: higher precision) (length: nr_values)
//          nr_values:      number of points where polynomial is evaluated
// output:
//          yvalues:        values of evaluated polynomial (length: nr_values)
void polyval(float* coeffs, int nr_coeffs, double* xvalues, int nr_values, double* yvalues);

// find values larger than 0 (by epsilon)
// input:
//          a:              array of values to check (length: len)
//          len:            number of values to check 
// output:
//          non_zero:       array with indices of values larger than 0 (by epsilon) (length: len)
int find_nonzero(double* a, int len, int* non_zero);

// find values at special indices 
// input:
//          a:              array from which specific values are asked (length: >= nr_indices)
//          indices:        array with indices (length: nr_indices)  must be ordered (ascending) and the last index must be <= length(a)
//          nr_indices:     number of indices
// output:
//          a:              overwrite array a with the values at the indices
void values_at_indices(double* a, int* indices, int nr_indices);

// itialize the identity matrix
// input:
//          a:              array (dim nxn) (row major layout)
//          n:              size of a
// output:
//          a:              overwrite array a with identity matrix
void init_identity(float* a, int n);
void init_identity(std::complex<float>* a, int n);

// itialize zero matrix
// input:
//          a:              array (dim mxn) (row major layout)
//          m:              number of rows
//          n:              number of columns
// output:
//          a:              overwrite array a with zeros
void init_zeros(float* a, int m, int n);
void init_zeros(std::complex<float>* a, int m, int n);

// copy one matrix to another of the same size (both in row major layout)
// input:
//          a:              matrix (dim mxn) (row major layout)
// output:
//          b:              overwrite array b with entries in a (dim mxn)
void copy_matrix(float* a, int m, int n, float* b);
void copy_matrix(std::complex<float>* a, int m, int n, std::complex<float>* b);

// subtract the mean of the rows of matrix a
// input:
//          a:              matrix (dim nxT) (row major layout)
//          lda:            leading dimension of a (>=T)
//          n:              number of rows
//          T:              number of columns
// output:
//          a:              overwrite array a with its rows of zero mean
void subtract_mean(float* a, int lda, int n, int T);
void subtract_mean(std::complex<float>* a, int lda, int n, int T);

// compute the mean of array a
float compute_mean(float* a, int len);
std::complex<float> compute_mean(std::complex<float>* a, int len);

// normalize array a (to Euclidean length 1)
void normalize(float* a, int len);
void normalize(complex<float>* a, int len);

// compute Euclidean norm of array a
float compute_norm(float* a, int len);
float compute_norm(complex<float>* a, int len);

// Auxiliary routine: printing a matrix (print maximally a 5x5 matrix)
// input:
//          desc:           name of matrix to be printed
//          a:              matrix (dim nxn) (row major layout)
//          lda:            leading dimension of a (>=n)
//          m:              number of rows
//          n:              number of columns
void print_matrix(const char* desc, int m, int n, float* a, int lda);
void print_matrix(const char* desc, int m, int n, double* a, int lda);
void print_matrix(const char* desc, int m, int n, std::complex<float>* a, int lda);

#endif