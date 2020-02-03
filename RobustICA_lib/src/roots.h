#ifndef ROOTS_H_
#define ROOTS_H_

#include <complex> // for complex numbers

using namespace std;

// Find the roots of a polynomial
// input:
//          coeffs:         coefficients of the polynomial
//          nr_coeffs:      number of coefficients (= degree of polynomial + 1) y = coeffs[0]*x^(nr_coeffs-1)+...
// output:
//          wr:             real parts of the found sources (size: nr_coeffs-1)
// for complex: 
//          r:              complex roots
void roots(float* coeffs, int nr_coeffs, double* wr);
void roots(complex<float>* coeffs, int nr_coeffs, complex<double>* r);

#endif