#ifndef PCA_H_
#define PCA_H_

#include <complex> // for complex numbers

using namespace std;

// PCA of the matrix a (used as prewhitening for BSS)
// input:
//          a:              matrix on which PCA is performed (dim mxn) (row major layout)
//          lda:            leading dimension of a (>=n)
//          m:              number of rows of a (= number of observations (if used as prewhitening))
//          n:              number of columns of a (= length of observations)
// output:
//          a:              a is overwritten with the principal vectors (= right singular vectors)
void PCA(float* a, int lda, int m, int n);
void PCA(complex<float>* a, int lda, int m, int n);

#endif