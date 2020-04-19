This document enumerates arrayfire API across different language wrappers.
They are grouped into the following sections.

- [Array Creation](#array-creation)
- [Fetch Array Information](#fetch-array-info)
- [Indexing Operations](#indexing)
- [Reshaping Array Data](#array-reshape)
- [Matrix Math](#matrix-math)
- [Statistics](#statistics)
- [Basic Math](#basic-math)
- [Signal Processing](#signal-processing)

### Array Creation
|                              | ArrayFire (C++)            | ArrayFire (Python)                         |
| :--------------------------- | :------------------------: | :----------------------------------------: |
| Empty array                  | array A(rows, cols, f64)   | af.Array(A, (rows, cols),af.Dtype)         |
| Identity matrix              | identity(rows, cols)       | af.identity(rows, cols)                    |
| Array filled with constant n | constant(n, rows, cols)    | af.constant(n, rows, cols)                 |
| Randomly filled array        | randu(rows, cols)          | af.randu(rows, cols)                       |
| Diagonal vector to matrix    | diag(A)                    | af.diag(af.range(n), num=0, extract=False) |

### Fetch Array Info
|                           | ArrayFire (C++)     | ArrayFire (Python)    |
| :------------------------ | :-----------------: | :-------------------: |
| Total number of elements  | A.elements()        | A.elements()          |
| Number of dimensions      | A.ndims()           | A.ndims()             |
| Shape of matrix           | A.dims()            | A.dims()              |
| Number of rows            | A.dims(0)           | A.dims()[0]           |
| Number of columns         | A.dims(1)           | A.dims()[1]           |

### Indexing
|                                | ArrayFire (C++)                            | ArrayFire (Python)                 |
| :----------------------------- | :----------------------------------------: | :--------------------------------: |
| Sequences                      | af::seq(low,high,step)                     |                                    |
| **Vector**                     |                                            |                                    |
| i+1 element                    | A(i)                                       | A[i]                               |
| last element                   | A(end)                                     | A                                  |
|                                | A(-1)                                      | A[-1]                              |
| 1 to i elements                | A(seq(i))                                  | A[:i]                              |
| i+1 to END elements            | A(seq(i, end))                             | A[i:]                              |
| i1+1 to i2+1 elements          | A(seq(i1, i2))                             | A[i1:i2+1]                         |
| i1+1 to i2+1 elements by step  | A(seq(i1, i2, step))                       | A[i1:i2+1:step]                    |
| **Matrix**                     |                                            |                                    |
| i+1 element                    | A(i)                                       |                                    |
| last element                   | A(end)                                     |                                    |
|                                | A(-1)                                      | A[-1]                              |
| Element(i+1,j+1)               | A(i, j)                                    | A[i, j]                            |
| i+1 row                        | A(i, span)                                 | A[i, :]                            |
| i+1 column                     | A(span, i)                                 | A[:, i]                            |
|                                | A(seq(i, end), seq(j, end))                |                                    |
|                                | A(seq(i, end), span)                       | A[i:, :]                           |
|                                | A(seq(i1, i2), seq(j1, j2))                | A[i1:i2+1, j1:j2+1]                |
|                                | A(seq(i1, i2, step1),seq(j1, j2, step2))   | A[i1:i2+1:step1, j1:j2+1:step2]    |
| i+1 row                        | A.row(i)                                   | A[i,:]                             |
| i+1 to j+1 rows                | A.rows(i, j)                               | A[i:j+1, :]                        |
| i+1 column                     | A.col(i)                                   | A[:, i]                            |
| i+1 to j+1 columns             | A.cols(i, j)                               | A[:, i:j+1]                        |
| i+1 slice/matrix               | A.slice(i)                                 | A[:, :, i]                         |
| i+1 to j+1 slices              | A.slices(i, j)                             | A[:, :, i:j+1]                     |

### Array Reshape
|                              | ArrayFire (C++)            | ArrayFire (Python)         |
| :--------------------------- | :------------------------: | :------------------------: |
| Flatten to vector            | flat(A)                    | af.flat(A)                 |
| Reshape Dimensions           | moddims(A, rows, cols)     | af.moddims(A, rows, cols)  |
| Conjugate Transposition      | A.H()                      | A.H                        |
| Non-conjugate transpose      | A.T()                      | A.T                        |
| Conjugate of complex values  | conjg(A)                   | af.conjg                   |
| Flip left to right           | flip(A, 0)                 | af.flip(A)                 |
| Flip up to down              | flip(A, 1)                 | af.flip(A, 1)              |
| Repeat matrix                | tile(A, i, j)              | af.tile(A, i, j)           |
| Swap axis                    | reorder(A, 1, 0)           | af.reorder(A, 1, 0)        |
| Bind columns                 | join(1, A1, A2)            | af.join(1, A1, A2)         |
| Bind rows                    | join(0, A1, A2)            | af.join(0, A1, A2)         |
| Shift                        | shift(A, 1)                | af.shift(A, 1)             |

### Matrix Math
|                           | ArrayFire (C++)              | ArrayFire (Python)                   |
| :------------------------ | :--------------------------: | :----------------------------------: |
| Upper Triangular Matrix   | lower(A)                     | af.lower(A)                          |
| Lower Triangular Matrix   | upper(A)                     | af.upper(A)                          |
| Rank                      | rank(A)                      |                                      |
| Determinant               | det\<float\>(A)                | af.det(A)                            |
| Euclid  norm              | norm(A)                      | af.norm(A, af.NORM.EUCLID)           |
| L1 norm                   | norm(A, AF_NORM_VECTOR_1)    | af.norm(A, af.NORM.MATRIX.1)         |
| L2 norm                   | norm(A, AF_NORM_VECTOR_2)    | af.norm(A, af.NORM.MATRIX.2)         |
| L4 norm                   | norm(A, AF_NORM_VECTOR_P_4)  | af.norm(A, af.NORM.MATRIX.L.PQ,4)    |
| L.inf norm                | norm(A, AF_NORM_VECTOR_INF)  | af.norm(A, af.NORM.MATRIX.INF)       |
| Solve Equation            | solve(A, B)                  | af.solve(A, B)                       |
| Inverse                   | inverse(A)                   | af.inverse(A)                        |
| LU                        | lu(out, pivot, A)            | L,U,P=af.lu(A)                       |
| Cholesky factorization    | cholesky(R, A, true)         | R,info=af.cholesky(A)                |
| QR                        | qr(Q, R, tau, A)             | Q,R,T=af.qr(A)                       |
| Singular values           | svd(U, s, Vt, A)             | U,s,Vt = af.svd(A)                   |
| Addition                  | A1 + A2                      | A1 + A2                              |
| Subtraction               | A1 - A2                      | A1 - A2                              |
| Multiplication            | A1 * A2                      | A1 * A2                              |
| Division                  | A = B / C                    | A = B / C                            |
| Matrix Multiplication     | matmul(A1, A2)               | af.matmul(A1,A2)                     |
| Dot Product(vector)       | dot(x1, x2)                  | af.dot(x1,x2)                        |

### Statistics
|                                   | ArrayFire (C++)          | ArrayFire (Python)    |
| :-------------------------------- | :----------------------: | :-------------------: |
| Sums of columns                   | sum(A)                   | af.sum(A, 0)          |
| Sums of rows                      | sum(A, 1)                | af.sum(A, 1)          |
| Sums of all elem.                 | sum\<float\>(A)            | af.sum(A)             |
| Products of columns               | product(A)               | af.product(A, 0)      |
| Products of rows                  | product(A, 1)            | af.product(A, 1)      |
| Products of all elem.             | product\<float\>(A)        | af.product(A)         |
| Averages of columns               | mean(A)                  | af.mean(A, dim=0)     |
| Averages of rows                  | mean(A, 1)               | af.mean(A, dim=1)     |
| Averages of all elem.             | mean\<float\>(A)           | af.mean(A)            |
| Maximum of columns                | max(A)                   | af.max(A, dim=0)      |
| Maximum of rows                   | max(A, 1)                | af.max(A, dim=1)      |
| Maximum of all elem.              | max\<float\>(A)            | af.max(A)             |
| Minimum of columns                | min(A)                   | af.min(A, dim=0)      |
| Minimum of rows                   | min(A, 1)                | af.min(A, dim=1)      |
| Minimum of all elem.              | min\<float\>(A)            | af.min(A)             |
| Variance  of columns              | var(A, 1, 0)             |                       |
| Variance  of rows                 | var(A, 1, 1)             |                       |
| Variance s of all elem.           | var\<float\>(A)            | var(A)                |
| Stand. deviations of columns      | stdev(A)                 |                       |
| Stand. deviations of rows         | stdev(A, 1)              |                       |
| Stand. deviations of all elem.    | stdev\<float\>(A)          | af.stdev(A)           |

### Basic Math
|                     | ArrayFire (C++)   | ArrayFire (Python)   |
| :------------------ | :---------------: | :------------------: |
| Round               | round(A)          | af.round(A)          |
| Round down          | floor(A)          | af.floor(A)          |
| Round up            | ceil(A)           | af.ceil(A)           |
| Exponential         | exp(A)            | af.exp(A)            |
| Element-wise power  | pow(A, 2)         | af.pow(A, 2)         |

### Signal Processing
|                        | ArrayFire (C++)   | ArrayFire (Python)   |
| :--------------------- | :---------------: | :------------------: |
| FFT1D (each column)    | fft(A)            | af.fft(A)            |
| FFT1D (each row)       | fft(A.T).T        | af.fft(A.T).T        |
