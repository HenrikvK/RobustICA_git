#ifndef ROBUSTICA_H_
#define ROBUSTICA_H_

#include <complex> // for complex numbers

using namespace std;

// RobustICA method to separate (real-valued) sources from a mixture of (real-valued) observations
// input:           
//      a:          matrix of observations (dim: nxT)
//      lda:        leading dimension of a (for row major layout) >=T
//      n:          number of observations = number of sources
//      T:          lengths of source/observation signals
//      kurtsign:   array of length n with the kurtosis sign of the sources to be searched for (+1 or -1) (or 0)                
//      prewhite:   if prewhitening is to be done on the signals
//      Wini:       extracting vectors initialization for RobustICA iterative search;
//                  if empty, identity matrix of suitable dimensions is used
//                  default advice: identity matrix of (dim nxn) (row major layout)
//      maxiter:    default advice: 1000
// output:
//      S:          matrix with the found sources (dim nxT) (row major layout)
// defaults:
//      deftype     = orthogonalization (no regression implemented)
void RobustICA(float* a, int lda, int n, int T, int* kurtsign, bool prewhite, float* Wini, int maxiter, float* S);
void RobustICA(complex<float>* a, int lda, int n, int T, int* kurtsign, bool prewhite, complex<float>* Wini, int maxiter, complex<float>* S);

// Computes optimal step size in the gradient - based optimization of the normalized kurtosis contrast (single iteration).
// output:
//      g:          search direction(normalized gradient vector)
//      mu_op:      optimal step size globally optimizing the normalized kurtosis contrast function
//      norm_g:     non-normalized gradient vector norm.
// input: 
//      w:          current extracting vector coefficients (length: L)
//      X:          sensor - output data matrix(one signal per row, one sample per column) (dim: LxT) (row major layout)
//      T:          length of time signals
//      s:          source kurtosis sign; if zero, the maximum absolute value of the contrast is sought
//      P:          projection matrix(used in deflationary orthogonalization; identity matrix otherwise) (dim: LxL)(row major layout)
//for complex only:
//      wreal:      if different from zero, keep extracting vector real valued by retaining only the
void kurt_gradient_optstep(float* w, int L, float* X, int T, int s, float* P, float* g, float& mu_opt, float& norm_g);
void kurt_gradient_optstep(complex<float>* w, int L, complex<float>* X, int T, int s, complex<float>* P, int wreal, complex<float>* g, float& mu_opt, float& norm_g);

#endif