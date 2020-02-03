## Kurtosis-based RobustICA method for deflationary ICA/BSS in C++.
This repository entails the C++ implementation of the RobustICA method developed by Zarzoso et. al

### REFERENCE: <br>
V. Zarzoso and P. Comon, "Robust Independent Component Analysis by Iterative Maximization of the Kurtosis Contrast with Algebraic Optimal Step Size", IEEE Transactions on Neural Networks, Vol. 21, No. 2, February 2010, pp. 248-261. <br>
See also the [Matlab version from Zarzoso et al.](http://www.i3s.unice.fr/~zarzoso/robustica.html)

### LIBRARIES:
The code requires the following libraries that have to be installed: 
* MKL (MATH KERNEL LIBRARY) from Intel <br>
    To rewrite the RobustICA method without MKL, one needs to write custom code for:
    1. Principal component analysis (or singular value decomposition) (real and complex)
    2. Function to solve an eigenvalue problem (or other method to find roots of a polynomial) (real and complex)
    3. Matrix-matrix multiplications (real and complex)
    4. Elementwise vector-vector multiplications (real and complex)

### CODE:
1. RobustICA.h/.cpp <br>
    * RobustICA method to extract Source matrix S <br>
        Works for real (float*) or complex (complex<float>*) observations
2. PCA.h/.cpp
    Principal Component Analysis which can be used separately or as prewhitening for ICA
    * For real or complex signals
    * Uses MKL Singular Value Decomposition function
3. roots.h/.cpp
    * Find the roots of a polynomial by constructing a companion matrix and 
    solving the corresponding eigenvalue problem
4. matrix_utils.h/.cpp
    * Useful matrix operations needed for ICA

### LIMITATIONS:
Compared to the [Matlab version from Zarzoso et al.](http://www.i3s.unice.fr/~zarzoso/robustica.html), the implementation does not include the following options:
1. to return the mixing matrix H
2. 'deftype' = regression <br>
     only: 'deftype' = orthogonalization
3. 'dimred'  <br>
     no dimensionality reduction (=> number of observations = number of sources)

These additional features can, however, be easily added to the code

#### ADDITIONAL REMARKS:
1. All matrices are saved in row major layout <br> 
    the leading dimension of an array is the number of elements after which the next row starts (usually the same as the column number)


