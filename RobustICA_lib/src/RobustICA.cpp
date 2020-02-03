#include "RobustICA.h"
#include <complex> // for complex numbers
#include <algorithm> // for minimum
#include "matrix_utils.h"
#include "PCA.h"
#include "roots.h"

// keep include/define in that order
#include <mkl_types.h>
#define MKL_Complex8 std::complex<float> 
#define MKL_Complex16 std::complex<double> // calculate roots in higher accuracy
#include <mkl.h>

using namespace std;

void RobustICA(float* a, int lda, int n, int T, int* kurtsign, bool prewhite, float* Wini, int maxiter, float* S) {
    bool verbose = false;    // set to true to plot information
    float tol = (float) 1e-3;       //tolerance for early stopping

    // subtract mean from observations: 
    subtract_mean(a, lda, n, T);

    if (prewhite)
        PCA(a, lda, n, T);

    //  number of remaining observations (may change under dimensionality reduction)
    // int dimobs = n;

    float* W = (float*)malloc(n * n * sizeof(float));      // extracting vectors
    init_zeros(W, n, n);
    float* buf_w = (float*)malloc(n * sizeof(float));     // create a buffer for w
    float* P = (float*)malloc(n * n * sizeof(float));      // projection matrix for deflationary orthogonalization(if required)
    init_identity(P, n);                                 // initialize P as identity matrix:

    tol = tol / sqrt((float)T);            // a statistically - significant termination threshold
    float tol2 = (tol * tol) / 2;     // the same threshold in terms of extracting vectors' absolute scalar product
    int* iter = (int*)malloc(n * sizeof(int));  // number of iterations

    // memory allocation
    float* g = (float*)malloc(n * sizeof(float)); 
    float* wg = (float*)malloc(n * sizeof(float)); 
    float* wn = (float*)malloc(n * sizeof(float));
    float* W2 = (float*)malloc(n * n * sizeof(float));
    // iterate over all sources
    for (int k = 0; k < n; k++) {// watch out: k = 1:n
        if (verbose)  printf("Source %d\n", k);

        int it = 0;
        bool keep_going = true;

        float* w = Wini + n * k;   // initialization
        normalize(w, n);            // normalization

        // project onto extracted vectors' orthogonal subspace (if deflationary orthogonalization)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, 1, n, 1, P, n, w, 1, 0, buf_w, 1); // buf_w = P * w;
        for (int i = 0; i < n; i++) { w[i] = buf_w[i]; }// copy buf_w back to w

        int signkurt = kurtsign[k];

        // iterate to extract one source
        while (keep_going) {
            it = it + 1;
            // compute KM optimal step size for gradient descent
            float mu_opt = 0;                               // default optimal step - size value
            float norm_g = 0;                               // initialize gradient norm
            kurt_gradient_optstep(w, n, a, T, signkurt, P, g, mu_opt, norm_g);

            // update extracting vector
            // wn = P * (w + mu_opt * g);
            for (int i = 0; i < n; i++) {
                wg[i] = w[i] + mu_opt * g[i]; // add gradient step
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, 1, n, 1, P, n, wg, 1, 0, wn, 1); //project onto P: wn = P * wg; dim P = nxn; dim wg = nx1; dim wn = nx1
            normalize(wn, n);

            // extracting vector convergence test
            float wnw;
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                1, 1, n, 1, wn, 1, w, 1, 0, &wnw, 1); //  wnw  = wn'*w dim wn = nx1; dim w = nx1, dim wnw = 1x1
            float th = abs(1 - abs(wnw));

            // copy wn back to w:
            for (int i = 0; i < n; i++)
                w[i] = wn[i];

            if (th < tol2 || norm_g < tol || it >= maxiter || mu_opt == 0) {
                // finish when extracting vector converges, the gradient is too small,
                // too many iterations have been run, or the optimal step - size is zero
                keep_going = 0;
            }
        }

        // estimated source
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            1, T, n, 1, w, 1, a, T, 0, S + k * T, T); //s = w'*a; dim w = nx1; dim a=nxT; dim s = 1xT 

        iter[k] = it;  // number of  iterations

        // copy w in W:
        for (int i = 0; i < n; i++)
            W[i * n + k] = w[i];

        //P = I - W * W';   // projection matrix for orthogonalization (if required)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            n, n, n, 1, W, n, W, n, 0, W2, n);   //W2 = W*W'; dim W = nxn; dim W2 = nxn

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    P[i * n + j] = 1 - W2[i * n + j];
                }
                else {
                    P[i * n + j] = -W2[i * n + j];
                }
            }
        }

    }
    std::free(W);
    std::free(buf_w);
    std::free(P);
    std::free(iter);   
    std::free(g);
    std::free(wg);
    std::free(wn);
    std::free(W2);
}
void RobustICA(complex<float>* a, int lda, int n, int T, int* kurtsign, bool prewhite, complex<float>* Wini, int maxiter, complex<float>* S) {
    bool verbose = false;    // set to true to plot information
    float tol = (float) 1e-3;       //tolerance for early stopping

    // subtract mean from observations: 
    subtract_mean(a, lda, n, T);

    if (prewhite)
        PCA(a, lda, n, T);

    //  number of remaining observations (may change under dimensionality reduction)
    // int dimobs = n;

    MKL_Complex8* W = (MKL_Complex8*)malloc(n * n * sizeof(MKL_Complex8));      // extracting vectors
    init_zeros(W, n, n);
    MKL_Complex8* buf_w = (MKL_Complex8*)malloc(n * sizeof(MKL_Complex8));     // create a buffer for w
    MKL_Complex8* P = (MKL_Complex8*)malloc(n * n * sizeof(MKL_Complex8));      // projection matrix for deflationary orthogonalization(if required)
    init_identity(P, n);                                 // initialize P as identity matrix:

    tol = tol / sqrt((float)T);            // a statistically - significant termination threshold
    float tol2 = (tol * tol) / 2;     // the same threshold in terms of extracting vectors' absolute scalar product
    int* iter = (int*)malloc(n * sizeof(int));  // number of iterations

    // memory allocation
    MKL_Complex8* g = (MKL_Complex8*)malloc(n * sizeof(MKL_Complex8));  
    MKL_Complex8* wg = (MKL_Complex8*)malloc(n * sizeof(MKL_Complex8)); 
    MKL_Complex8* wn = (MKL_Complex8*)malloc(n * sizeof(MKL_Complex8));
    MKL_Complex8* W2 = (MKL_Complex8*)malloc(n * n * sizeof(MKL_Complex8));
    // iterate over all sources
    for (int k = 0; k < n; k++) {// watch out: k = 1:n
        if (verbose) { printf("Source %d\n", k); }

        int it = 0;
        bool keep_going = true;

        MKL_Complex8* w = Wini + n * k;     // initialization
        normalize(w, n);                    // normalization

        MKL_Complex8 alpha = complex<float>(1, 0);
        MKL_Complex8 beta = complex<float>(0, 0);
        // project onto extracted vectors' orthogonal subspace (if deflationary orthogonalization)
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, 1, n, &alpha, P, n, w, 1, &beta, buf_w, 1); // buf_w = P * w;
        for (int i = 0; i < n; i++) { w[i] = buf_w[i]; }// copy buf_w back to w

        int signkurt = kurtsign[k];

        // iterate to extract one source
        while (keep_going) {
            it = it + 1;
            // compute KM optimal step size for gradient descent
            float mu_opt = 0;                               // default optimal step - size value
            float norm_g = 0;                               // initialize gradient norm
            int wreal = 0;
            kurt_gradient_optstep(w, n, a, T, signkurt, P, wreal, g, mu_opt, norm_g);

            // update extracting vector
            // wn = P * (w + mu_opt * g);
            for (int i = 0; i < n; i++) {
                wg[i] = w[i] + MKL_Complex8(mu_opt, 0) * g[i]; // add gradient step
            }
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, 1, n, &alpha, P, n, wg, 1, &beta, wn, 1); //project onto P: wn = P * wg; dim P = nxn; dim wg = nx1; dim wn = nx1
            normalize(wn, n);

            // extracting vector convergence test
            MKL_Complex8 wnw;
            cblas_cgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
                1, 1, n, &alpha, wn, 1, w, 1, &beta, &wnw, 1); //  wnw  = wn'*w dim wn = nx1; dim w = nx1, dim wnw = 1x1
            float th = abs(1 - abs(wnw));

            // copy wn back to w:
            for (int i = 0; i < n; i++)
                w[i] = wn[i];

            if (th < tol2 || norm_g < tol || it >= maxiter || mu_opt == 0) {
                // finish when extracting vector converges, the gradient is too small,
                // too many iterations have been run, or the optimal step - size is zero
                keep_going = 0;
            }
        }

        // estimated source
        cblas_cgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
            1, T, n, &alpha, w, 1, a, T, &beta, S + k * T, T); //s = w'*a; dim w = nx1; dim a=nxT; dim s = 1xT 
        iter[k] = it;  // number of  iterations

        // copy w in W:
        for (int i = 0; i < n; i++)
            W[i * n + k] = w[i];

        //P = I - W * W';   // projection matrix for orthogonalization (if required)
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
            n, n, n, &alpha, W, n, W, n, &beta, W2, n);   //W2 = W*W'; dim W = nxn; dim W2 = nxn

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    P[i * n + j] = MKL_Complex8(1, 0) - W2[i * n + j];
                }
                else {
                    P[i * n + j] = -W2[i * n + j];
                }
            }
        }

    }
    std::free(W);
    std::free(buf_w);
    std::free(P);
    std::free(iter);    
    std::free(g);
    std::free(wg);
    std::free(wn);
    std::free(W2);
}

void kurt_gradient_optstep(float* w, int L, float* X, int T, int s, float* P, float* g, float& mu_opt, float& norm_g) {
    bool verbose = false;
    mu_opt = 0; // default optimal step - size value
    norm_g = 0; // initialize gradient norm

    // Compute search direction(gradient vector)
    // compute necessary interim values
    float* y = (float*)malloc(T * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        1, T, L, 1, w, 1, X, T, 0, y, T); //y = w'*X; dim w= Lx1, dim X= LxT, dim y=1xT

    float* ya2 = (float*)malloc(T * sizeof(float));
    vsMul(T, y, y, ya2);        //ya2 = y.*conj(y);  WATCH OUT for complex: Should be conjugate!
    float* y2 = (float*)malloc(T * sizeof(float));
    vsMul(T, y, y, y2);         //y2 = y.*y;
    float* ya4 = (float*)malloc(T * sizeof(float));
    vsMul(T, ya2, ya2, ya4);    //ya4 = ya2.*ya2;

    float Eya2 = compute_mean(ya2, T);
    float Ey2 = compute_mean(y2, T);
    float Eya4 = compute_mean(ya4, T);

    float epsf = std::numeric_limits<float>::epsilon();
    if (abs(Eya2) < epsf) { // check for zero denominator
        if (verbose) { printf(">>> OPT STEP SIZE: zero power\n"); }
        for (int i = 0; i < L; i++) {
            g[i] = 0;       // set gradient to zero. 
        }
        norm_g = 0;
    }
    else {
        // compute gradient if contrast denominator is not null
        float* Eycx = (float*)malloc(L * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            L, 1, T, (float)(1.0 / (float)T), X, T, y, T, 0, Eycx, 1); // WATCH OUT for complex: same as Eyx: check again for complex!
        float* Eyx = (float*)malloc(L * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            L, 1, T, (float)(1.0 / (float)T), X, T, y, T, 0, Eyx, 1);  // Eycx = X * y' /T   | dim X = LxT, dim y= 1xT, dim Eyx = Lx1
        float* yya2 = (float*)malloc(T * sizeof(float));
        vsMul(T, ya2, y, yya2);                                 //yya2 = ya2.*y
        float* Ey3x = (float*)malloc(L * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            L, 1, T, (float)(1.0 / (float)T), X, T, yya2, T, 0, Ey3x, 1);// Ey3x = X * yya2' /T | dim X = LxT, dim yya2= 1xT, dim Ey3x = Lx1
        // contrast numerator and denominator at current point
        float p1 = Eya4 - abs(Ey2) * abs(Ey2);
        float p2 = Eya2;

        float* g_buf = (float*)malloc(L * sizeof(float));
        for (int i = 0; i < L; i++) {  //g = 4 * ((Ey3x - Eyx * Ey2')*p2 - p1*Eycx )/p2^3;
            g_buf[i] = (float) (4.0 * ((Ey3x[i] - Eyx[i] * Ey2) * p2 - p1 * Eycx[i]) / (float)(p2 * p2 * p2));
        }

        // project if required  (normalize later)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            L, 1, L, 1, P, L, g_buf, 1, 0, g, 1);   // g = P * g; dim g = LxL, dim g = Lx1
        norm_g = compute_norm(g, L);                // norm_g = norm(g);

        if (norm_g < epsf) {
            if (verbose) { printf(">>> OPT STEP SIZE: zero gradient\n"); }
        }
        else {
            // if (wreal) { g = real(g); }  // WATCH OUT for complex ONLY FOR COMPLEX

            normalize(g, L);    // normalize the gradient->parameter of interest : direction
                                // improves conditioning of opt step - size polynomial

            // Compute optimal step size
            float* gg = (float*)malloc(T * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                1, T, L, 1, g, 1, X, T, 0, gg, T); // gg = g'*X; dim g = Lx1, dim X = LxT, dim gg = 1xT

            // calculate interim values for contrast rational function
            //ya2 = y.*conj(y);         // Does not need to be calculated again!
            float* g2 = (float*)malloc(T * sizeof(float));
            vsMul(T, gg, gg, g2);       // g2 = gg.*gg

            float* yg = (float*)malloc(T * sizeof(float));
            vsMul(T, y, gg, yg);       // yg = y.*gg;    

            // WATCH out: additonal for complex: //ga2 = gg.*conj(gg); //reygc = real(y.*conj(gg));
            //Eya2reygc = mean(ya2.*reygc);
            //Ereygc2     = mean(reygc. ^ 2);
            //Ega2reygc   = mean(ga2.*reygc);
            // Ega4        = mean(ga2. ^ 2);
            // Eya2ga2     = mean(ya2.*ga2);
            // Ega2        = mean(ga2);      // Extra for complex
            // Ereygc      = mean(reygc);   // Extra for complex

            // only for the real case: 
            float* ya2yg = (float*)malloc(T * sizeof(float));
            vsMul(T, ya2, yg, ya2yg);
            float Eya2reygc = compute_mean(ya2yg, T);       //Eya2reygc = mean(ya2.*reygc);
            float* reygc2 = (float*)malloc(T * sizeof(float));
            vsMul(T, yg, yg, reygc2);
            float Ereygc2 = compute_mean(reygc2, T);        //Ereygc2     = mean(reygc. ^ 2);
            float* ga2reygc = (float*)malloc(T * sizeof(float));
            vsMul(T, g2, yg, ga2reygc);
            float Ega2reygc = compute_mean(ga2reygc, T);    //Ega2reygc   = mean(ga2.*reygc);
            float* g4 = (float*)malloc(T * sizeof(float));
            vsMul(T, g2, g2, g4);
            float Ega4 = compute_mean(g4, T);               // Ega4 = mean(ga2. ^ 2);
            float* ya2g2 = (float*)malloc(T * sizeof(float));
            vsMul(T, ya2, g2, ya2g2);
            float Eya2ga2 = compute_mean(ya2g2, T);         //Eya2ga2 = mean(ya2.*ga2);

            float Eg2 = compute_mean(g2, T);     // Eg2 = mean(g2); 
            float Eyg = compute_mean(yg, T);     // Eyg = mean(yg);


            float h0 = Eya4 - abs(Ey2) * abs(Ey2);
            float h1 = 4 * Eya2reygc - 4 * (Ey2 * Eyg); // different for complex
            float h2 = 4 * Ereygc2 + 2 * Eya2ga2 - 4 * abs(Eyg) * abs(Eyg) - 2 * (Ey2 * Eg2);// different for complex
            float h3 = 4 * Ega2reygc - 4 * (Eg2 * Eyg);
            float h4 = Ega4 - abs(Eg2) * abs(Eg2);

            float P_2[5] = { h4,  h3, h2,h1,h0 };        // P = [h4, h3, h2, h1, h0];

            float i0 = Eya2;
            float i1 = 2 * Eyg; //different for complex 
            float i2 = Eg2;

            float Q[3] = { i2, i1, i0 };

            // normalized kurtosis contrast = P / Q ^ 2 - 2
            float a0 = -2 * h0 * i1 + h1 * i0;
            float a1 = -4 * h0 * i2 - h1 * i1 + 2 * h2 * i0;
            float a2 = -3 * h1 * i2 + 3 * h3 * i0;
            float a3 = -2 * h2 * i2 + h3 * i1 + 4 * h4 * i0;
            float a4 = -h3 * i2 + 2 * h4 * i1;

            float p[5] = { a4, a3, a2, a1, a0 };
            double* rr = (double*)malloc(4 * sizeof(double));   // real parts of roots
            roots(p, 5, rr);    //rr = real(roots(p));       // keep real parts only

            double* Pval = (double*)malloc(4 * sizeof(double));        //Pval = polyval(P, rr);
            polyval(P_2, 5, rr, 4, Pval);
            double* Q2val = (double*)malloc(4 * sizeof(double));        //Q2val = polyval(Q, rr).^2;
            polyval(Q, 3, rr, 4, Q2val);
            for (int i = 0; i < 4; i++) {
                Q2val[i] = Q2val[i] * Q2val[i];
            }

            int* nonzero_Q2val = (int*)malloc(4 * sizeof(int));
            int nr_nonzero_Q2val = find_nonzero(Q2val, 4, nonzero_Q2val); //nonzero_Q2val = find(Q2val > eps); 
            // check roots not shared by denominator
            // NOTE: in theory, the denominator can never
            // cancel out if the gradient is used as a search direction, due to the orthogonality
            // between the extracting vectorand the corresponding gradient
            // (only exception : if it is the last source to be extracted;
            // but this scenario is detected by the gradient norm)

            if (nr_nonzero_Q2val == 0) {
                if (verbose) {
                    printf(">>> OPT STEP SIZE: all roots shared by denominator\n");
                    print_matrix("Pval", 1, 4, Pval, 4);
                    print_matrix("Q2val", 1, 4, Q2val, 4);
                }
            }
            else {
                values_at_indices(Pval, nonzero_Q2val, nr_nonzero_Q2val);  // Pval = Pval(nonzero_Q2val);
                values_at_indices(Q2val, nonzero_Q2val, nr_nonzero_Q2val); // Q2val = Q2val(nonzero_Q2val);
                values_at_indices(rr, nonzero_Q2val, nr_nonzero_Q2val);    // rr = rr(nonzero_Q2val);

                double* Jkm_val = (double*)malloc(nr_nonzero_Q2val * sizeof(double));
                vdDiv(nr_nonzero_Q2val, Pval, Q2val, Jkm_val); //Jkm_val = Pval. / Q2val - 2;// normalized kurtosis

                if (s) {        // maximize or minimize kurtosis value, depending on kurtosis sign
                    for (int i = 0; i < nr_nonzero_Q2val; i++) {
                        Jkm_val[i] = s * (Jkm_val[i] - 2);// Jkm_val = real(s * Jkm_val);
                                                        // watch out: only real value in case of cmplex signals.
                    }
                }
                else {          // maximize absolute kurtosis value, if no sign is given
                    for (int i = 0; i < nr_nonzero_Q2val; i++) {
                        Jkm_val[i] = abs(Jkm_val[i] - 2);
                    }
                }
                // find maximum
                double* maximum = std::max_element(Jkm_val, Jkm_val + nr_nonzero_Q2val);
                int loc = (int) std::distance(Jkm_val, maximum); // find location of minimum

                mu_opt = (float)rr[loc]; // optimal step size

                if (verbose) {
                    print_matrix("X", L, T, X, T);
                    print_matrix("w", 1, L, w, L);
                    print_matrix("Ey3x", 1, L, Ey3x, L);
                    print_matrix("Eyx", 1, L, Eyx, L);
                    print_matrix("g_buf", 1, L, g_buf, L);
                    print_matrix("Eycx", 1, L, Eycx, L);
                    printf("Ey2= %f \n", Ey2);
                    printf("p2 = %f \n", p2);
                    printf("p1 = %f \n", p1);
                    //print_matrix("g", L, 1, g, 1);
                    printf("norm_g = %f \n", norm_g);
                    print_matrix("g_normalized", L, 1, g, 1);
                    print_matrix("p", 1, 5, p, 5);
                    print_matrix("rr", 1, 4, rr, 4);
                    print_matrix("Pval", 1, 4, Pval, 4);
                    print_matrix("Q2val", 1, 4, Q2val, 4);
                    print_matrix("Jkm_val", 1, nr_nonzero_Q2val, Jkm_val, nr_nonzero_Q2val);
                }
                std::free(Jkm_val);
            }
            std::free(gg);
            std::free(g2);
            std::free(yg);
            std::free(ya2yg);
            std::free(reygc2);
            std::free(ga2reygc);
            std::free(g4);
            std::free(ya2g2);
            std::free(Pval);
            std::free(Q2val);
            std::free(rr);
            std::free(nonzero_Q2val);
        }
        std::free(Eycx);
        std::free(Eyx);
        std::free(yya2);
        std::free(Ey3x);
        std::free(g_buf);
    }
    std::free(y);
    std::free(ya2);
    std::free(y2);
    std::free(ya4);
}
void kurt_gradient_optstep(complex<float>* w, int L, complex<float>* X, int T, int s, complex<float>* P, int wreal, complex<float>* g, float& mu_opt, float& norm_g) {
    bool verbose = false;
    mu_opt = 0; // default optimal step - size value
    norm_g = 0; // initialize gradient norm

    // Compute search direction(gradient vector)
    // compute necessary interim values
    MKL_Complex8* y = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
    MKL_Complex8 alpha = complex<float>(1, 0);
    MKL_Complex8 beta = complex<float>(0, 0);
    cblas_cgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
        1, T, L, &alpha, w, 1, X, T, &beta, y, T); //y = w'*X; dim w= Lx1, dim X= LxT, dim y=1xT
     //print_matrix("y", 1, T, (complex<float>*) y, T);
     //print_matrix("X", 1, T, X, T);
     //print_matrix("X2", 1, T, (complex<float>*) X2, T);

    MKL_Complex8* ya2 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
    vcMulByConj(T, y, y, ya2);        //ya2 = y.*conj(y);  WATCH OUT for complex: Should be conjugate!
    MKL_Complex8* y2 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
    vcMul(T, y, y, y2);         //y2 = y.*y;
    MKL_Complex8* ya4 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
    vcMul(T, ya2, ya2, ya4);    //ya4 = ya2.*ya2;

    complex<float> Eya2 = compute_mean(ya2, T);
    complex<float> Ey2 = compute_mean(y2, T);
    complex<float> Eya4 = compute_mean(ya4, T);
    //print_matrix("ya4", 1, T, (complex<float>*) ya4, T);

    float epsf = std::numeric_limits<float>::epsilon();
    if (abs(Eya2) < epsf) { // check for zero denominator
        if (verbose) { printf(">>> OPT STEP SIZE: zero power\n"); }
        for (int i = 0; i < L; i++) {
            g[i] = complex < float>(0, 0);       // set gradient to zero. 
        }
        norm_g = 0;
    }
    else {
        // compute gradient if contrast denominator is not null
        MKL_Complex8 alpha2 = complex<float>((float)((1.0 / (float)T)), 0);
        MKL_Complex8 beta2 = complex<float>(0, 0);
        MKL_Complex8* Eycx = (MKL_Complex8*)malloc(L * sizeof(complex<float>));
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
            L, 1, T, &alpha2, X, T, y, T, &beta2, Eycx, 1);
        MKL_Complex8* Eyx = (MKL_Complex8*)malloc(L * sizeof(complex<float>));
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            L, 1, T, &alpha2, X, T, y, T, &beta2, Eyx, 1);  // Eycx = X * y' /T   | dim X = LxT, dim y= 1xT, dim Eyx = Lx1
        MKL_Complex8* yya2 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
        vcMul(T, ya2, y, yya2);                                 //yya2 = ya2.*y
        MKL_Complex8* Ey3x = (MKL_Complex8*)malloc(L * sizeof(complex<float>));
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
            L, 1, T, &alpha2, X, T, yya2, T, &beta2, Ey3x, 1);// Ey3x = X * yya2' /T | dim X = LxT, dim yya2= 1xT, dim Ey3x = Lx1
        // contrast numerator and denominator at current point
        complex<float> p1 = Eya4 - abs(Ey2) * abs(Ey2);
        complex<float> p2 = Eya2;

        //print_matrix("Eycx", 1, L, Eycx, L); 
        //print_matrix("Eyx", 1, L, Eyx, L);
        //print_matrix("Ey3x", 1, L, Ey3x, L);

        MKL_Complex8* g_buf = (MKL_Complex8*)malloc(L * sizeof(complex<float>));
        for (int i = 0; i < L; i++) {  //g = 4 * ((Ey3x - Eyx * Ey2')*p2 - p1*Eycx )/p2^3;
            g_buf[i] = complex<float>(4.0, 0) * ((Ey3x[i] - Eyx[i] * conj(Ey2)) * p2 - p1 * Eycx[i]) / (p2 * p2 * p2);
        }

        // project if required  (normalize later)
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            L, 1, L, &alpha, P, L, g_buf, 1, &beta, g, 1);   // g = P * g; dim g = LxL, dim g = Lx1
        //norm_g = compute_norm(g, L);                // norm_g = norm(g);
        int incr = 1;
        norm_g = scnrm2(&L, g, &incr);

        if (norm_g < epsf) {
            if (verbose) { printf(">>> OPT STEP SIZE: zero gradient\n"); }
        }
        else {
            // if (wreal) { g = real(g); }  // WATCH OUT for complex ONLY FOR COMPLEX

            // normalize the gradient->parameter of interest : direction
            // improves conditioning of opt step - size polynomial
            for (int i = 0; i < L; i++) {
                g[i] = g[i] / norm_g;
            }
            //print_matrix("g", 1, L, g, L);
            // Compute optimal step size
            MKL_Complex8* gg = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            cblas_cgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
                1, T, L, &alpha, g, 1, X, T, &beta, gg, T); // gg = g'*X; dim g = Lx1, dim X = LxT, dim gg = 1xT

            // calculate interim values for contrast rational function
            MKL_Complex8* ga2 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMulByConj(T, gg, gg, ga2);        // ga2 = gg.*conj(gg);
            for (int i = 0; i < T; i++) {
                ga2[i] = complex<float>(real(ga2[i]), 0);
            }

            MKL_Complex8* reygc = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMulByConj(T, y, gg, reygc);       // reygc = real(y.*conj(gg));
            for (int i = 0; i < T; i++) {
                reygc[i] = complex<float>(real(reygc[i]), 0);
            }
            MKL_Complex8* g2 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMul(T, gg, gg, g2);       // g2 = gg.*gg
            MKL_Complex8* yg = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMul(T, y, gg, yg);       // yg = y.*gg;    

            // only for the real case: 
            MKL_Complex8* ya2reygc = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMul(T, ya2, reygc, ya2reygc);
            MKL_Complex8 Eya2reygc = compute_mean(ya2reygc, T);       //Eya2reygc = mean(ya2.*reygc);
            MKL_Complex8* reygc2 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMul(T, reygc, reygc, reygc2);
            MKL_Complex8 Ereygc2 = compute_mean(reygc2, T);        //Ereygc2     = mean(reygc. ^ 2);
            MKL_Complex8* ga2reygc = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMul(T, ga2, reygc, ga2reygc);
            MKL_Complex8 Ega2reygc = compute_mean(ga2reygc, T);    //Ega2reygc   = mean(ga2.*reygc);
            MKL_Complex8* ga4 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMul(T, ga2, ga2, ga4);
            MKL_Complex8 Ega4 = compute_mean(ga4, T);               // Ega4 = mean(ga2. ^ 2);
            MKL_Complex8* ya2ga2 = (MKL_Complex8*)malloc(T * sizeof(complex<float>));
            vcMul(T, ya2, ga2, ya2ga2);
            MKL_Complex8 Eya2ga2 = compute_mean(ya2ga2, T);         //Eya2ga2 = mean(ya2.*ga2);
            MKL_Complex8 Ega2 = compute_mean(ga2, T);            // Ega2 = mean(ga2);   
            MKL_Complex8 Ereygc = compute_mean(reygc, T);       //Ereygc      = mean(reygc); 

            MKL_Complex8 Eg2 = compute_mean(g2, T);     // Eg2 = mean(g2); 
            MKL_Complex8 Eyg = compute_mean(yg, T);     // Eyg = mean(yg);

            MKL_Complex8 h0 = Eya4 - abs(Ey2) * abs(Ey2);
            MKL_Complex8 Eyg_conj = MKL_Complex8(real(Eyg), -imag(Eyg));
            MKL_Complex8 h1 = MKL_Complex8(4, 0) * Eya2reygc - MKL_Complex8(4, 0) * MKL_Complex8(real(Ey2 * Eyg_conj), 0); // different for complex
            MKL_Complex8 Eg2_conj = MKL_Complex8(real(Eg2), -imag(Eg2));
            MKL_Complex8 h2 = MKL_Complex8(4, 0) * Ereygc2 + MKL_Complex8(2, 0) * Eya2ga2 - MKL_Complex8(4, 0) * abs(Eyg) * abs(Eyg) - MKL_Complex8(2, 0) * MKL_Complex8(real(Ey2 * Eg2_conj), 0);// different for complex
            MKL_Complex8 h3 = MKL_Complex8(4, 0) * Ega2reygc - MKL_Complex8(4, 0) * MKL_Complex8(real(Eg2 * Eyg_conj), 0);
            MKL_Complex8 h4 = Ega4 - abs(Eg2) * abs(Eg2);

            float P_2[5] = { real(h4), real(h3), real(h2),real(h1),real(h0) };        // P = [h4, h3, h2, h1, h0];

            MKL_Complex8 i0 = Eya2;
            MKL_Complex8 i1 = MKL_Complex8(2, 0) * Ereygc;
            MKL_Complex8 i2 = Ega2;

            float Q[3] = { real(i2), real(i1), real(i0) };

            // normalized kurtosis contrast = P / Q ^ 2 - 2
            MKL_Complex8 a0 = MKL_Complex8(-2, 0) * h0 * i1 + h1 * i0;
            MKL_Complex8 a1 = MKL_Complex8(-4, 0) * h0 * i2 - h1 * i1 + MKL_Complex8(2, 0) * h2 * i0;
            MKL_Complex8 a2 = MKL_Complex8(-3, 0) * h1 * i2 + MKL_Complex8(3, 0) * h3 * i0;
            MKL_Complex8 a3 = MKL_Complex8(-2, 0) * h2 * i2 + h3 * i1 + MKL_Complex8(4, 0) * h4 * i0;
            MKL_Complex8 a4 = -h3 * i2 + MKL_Complex8(2, 0) * h4 * i1;

            MKL_Complex8 p[5] = { a4, a3, a2, a1, a0 };
            complex<double>* r = (complex<double>*)malloc(4 * sizeof(complex<double>));   //  roots
            roots((complex<float>*)p, 5, r);                     //rr = real(roots(p)); 
            double* rr = (double*)malloc(4 * sizeof(double));   //  roots
            for (int i = 0; i < 4; i++) {       // keep real parts only
                rr[i] = real(r[i]);
            }

            double* Pval = (double*)malloc(4 * sizeof(double));        //Pval = polyval(P, rr);
            polyval(P_2, 5, rr, 4, Pval);
            double* Q2val = (double*)malloc(4 * sizeof(double));        //Q2val = polyval(Q, rr).^2;
            polyval(Q, 3, rr, 4, Q2val);
            for (int i = 0; i < 4; i++) {
                Q2val[i] = Q2val[i] * Q2val[i];
            }

            int* nonzero_Q2val = (int*)malloc(4 * sizeof(int));
            int nr_nonzero_Q2val = find_nonzero(Q2val, 4, nonzero_Q2val); //nonzero_Q2val = find(Q2val > eps); 
            // check roots not shared by denominator
            // NOTE: in theory, the denominator can never
            // cancel out if the gradient is used as a search direction, due to the orthogonality
            // between the extracting vectorand the corresponding gradient
            // (only exception : if it is the last source to be extracted;
            // but this scenario is detected by the gradient norm)

            if (nr_nonzero_Q2val == 0) {
                if (verbose) {
                    printf(">>> OPT STEP SIZE: all roots shared by denominator\n");
                    print_matrix("Pval", 1, 4, Pval, 4);
                    print_matrix("Q2val", 1, 4, Q2val, 4);
                }
            }
            else {
                values_at_indices(Pval, nonzero_Q2val, nr_nonzero_Q2val);  // Pval = Pval(nonzero_Q2val);
                values_at_indices(Q2val, nonzero_Q2val, nr_nonzero_Q2val); // Q2val = Q2val(nonzero_Q2val);
                values_at_indices(rr, nonzero_Q2val, nr_nonzero_Q2val);    // rr = rr(nonzero_Q2val);

                double* Jkm_val = (double*)malloc(nr_nonzero_Q2val * sizeof(double));
                vdDiv(nr_nonzero_Q2val, Pval, Q2val, Jkm_val); //Jkm_val = Pval. / Q2val - 2;// normalized kurtosis

                if (s) {        // maximize or minimize kurtosis value, depending on kurtosis sign
                    for (int i = 0; i < nr_nonzero_Q2val; i++) {
                        Jkm_val[i] = s * (Jkm_val[i] - 2);// Jkm_val = real(s * Jkm_val);
                                                        // watch out: only real value in case of cmplex signals.
                    }
                }
                else {          // maximize absolute kurtosis value, if no sign is given
                    for (int i = 0; i < nr_nonzero_Q2val; i++) {
                        Jkm_val[i] = abs(Jkm_val[i] - 2);
                    }
                }
                // find maximum
                double* maximum = std::max_element(Jkm_val, Jkm_val + nr_nonzero_Q2val);
                int loc = (int) std::distance(Jkm_val, maximum); // find location of minimum

                mu_opt = (float)rr[loc]; // optimal step size

                if (verbose) {
                    print_matrix("Ey3x", 1, L, Ey3x, L);
                    print_matrix("Eyx", 1, L, Eyx, L);
                    print_matrix("g_buf", 1, L, g_buf, L);
                    print_matrix("Eycx", 1, L, Eycx, L);
                    printf("Ey2= %f \n", Ey2);
                    printf("p2 = %f \n", p2);
                    printf("p1 = %f \n", p1);
                    print_matrix("g", L, 1, g, 1);
                    printf("norm_g = %f \n", (double)norm_g);
                    print_matrix("g_normalized", L, 1, g, 1);
                    print_matrix("p", 1, 5, p, 5);
                    print_matrix("rr", 1, 4, rr, 4);
                    print_matrix("Pval", 1, 4, Pval, 4);
                    print_matrix("Q2val", 1, 4, Q2val, 4);
                    print_matrix("Jkm_val", 1, nr_nonzero_Q2val, Jkm_val, nr_nonzero_Q2val);
                    printf("mu_opt = %f \n", (double)mu_opt);
                }
                std::free(Jkm_val);
            }
            std::free(gg);
            std::free(ga2);
            std::free(reygc);
            std::free(g2);
            std::free(yg);
            std::free(ya2reygc);
            std::free(reygc2);
            std::free(ga2reygc);
            std::free(ga4);
            std::free(ya2ga2);
            std::free(r);
            std::free(rr);
            std::free(Pval);
            std::free(Q2val);
            std::free(nonzero_Q2val);
        }
        std::free(Eycx);
        std::free(Eyx);
        std::free(yya2);
        std::free(Ey3x);
        std::free(g_buf);
    }
    std::free(y);
    std::free(ya2);
    std::free(y2);
    std::free(ya4);
}

