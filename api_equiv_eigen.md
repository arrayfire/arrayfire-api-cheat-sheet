This document enumerates arrayfire API equivalent in Matlab,  Python (numpy,  scipy).
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
|                              | ArrayFire (C++)            | Eigen (C++)                    |
| :--------------------------- | :------------------------: | :----------------------------: |
| Identity matrix              | identity(rows, cols)       | MatrixXd::Identity(rows, cols) |
|                              |                            | C.setIdentity(rows, cols)      |
| n filled array               | constant(n, rows, cols)    |                                |
| 1 filled array               | constant(1, rows, cols)    | MatrixXd::Ones(rows, cols)     |
|                              |                            | C.setOnes(rows, cols)          |
| 0 filled array               | constant(0, rows, cols)    | MatrixXd::Zero(rows, cols)     |
|                              |                            | C.setZero(rows, cols)          |
| Random filled array          | randu(rows, cols)          | MatrixXd::Random(rows, cols)   |
|                              |                            | C.setRandom(rows, cols)        |
| Diagonal vector to matrix    | diag(R)                    | x.asDiagonal()                 |
|                              |                            | R.diagonal()                   |

### Fetch Array Info
|                              | ArrayFire (C++)            | Eigen (C++)                    |
| :--------------------------- | :------------------------: | :----------------------------: |
| Vector size                  | x.elements()               | x.size()                       |
| Number of dimensions         | R.ndims()                  |                                |
| Shape of matrix              | R.dims()                   |                                |
| Number of rows               | R.dims(0)                  | C.rows()                       |
| Number of columns            | R.dims(1)                  | C.cols()                       |
| Number of elements           | R.elements()               |                                |

### Indexing
|                               | ArrayFire (C++)                           | Eigen (C++)                            |
| :---------------------------  | :------------------------:                | :----------------------------:         |
| Sequences                     | af::seq(low, high, step)                  |                                        |
| **Vector**                    |                                           |                                        |
| i+1 element                   | x(i)                                      | x(i)                                   |
| last element                  | x(end)                                    | x.tail(1)                              |
|                               | x(-1)                                     |                                        |
| 1 to i elements               | x(seq(i))                                 |                                        |
| i+1 to END elements           | x(seq(i, end))                            |                                        |
| i1+1 to i2+1 elements         | x(seq(i1, i2))                            |                                        |
| i1+1 to i2+1 elements by step | x(seq(i1, i2, step))                      |                                        |
| **Matrix**                    |                                           |                                        |
| i+1 element                   | R(i)                                      |                                        |
| last element                  | R(end)                                    |                                        |
|                               | R(-1)                                     |                                        |
| Element(i+., j+1)             | R(i, j)                                   | C(i, j)                                |
| i+1 row                       | R(i, span)                                |                                        |
| i+1 column                    | R(span, i)                                | P.row(i)                               |
|                               | R(seq(i, end), seq(j, end))               | P.col(j)                               |
|                               | R(seq(i, end), span)                      |                                        |
|                               | R(seq(i1, i2), seq(j1, j2))               | P.block(i1,  j1,  rows,  cols)         |
|                               | R(seq(i1, i2, step1), seq(j1, j2, step2)) | P.rightCols\<cols\>()  P.rightCols(cols) |
|                               |                                           | P.leftCols\<cols\>() P.leftCols(cols)    |
| i+1 row                       | R.row(i)                                  | P.topRows\<rows\>() P.topRows(rows)      |
| i+1 to j+1 rows               | R.rows(i, j)                              | P.bottomRows\<rows\>()P.bottomRows(rows) |
| i+1 column                    | R.col(i)                                  |                                        |
| i+1 to j+1 columns            | R.cols(i, j)                              |                                        |
| i+1 slice                     | R3D.slice(i)                              |                                        |
| i+1 to j+1 slices             | R3D.slices(i, j)                          |                                        |

### Array Reshape
|                              | ArrayFire (C++)            | Eigen (C++)                       |
| :--------------------------- | :------------------------: | :----------------------------:    |
| Flatten to vector            | flat(R)                    |                                   |
| Reshaping                    | moddims(R, rows, cols)     | B.resize(rows, cols)              |
| Conjugate Transposition      | R.H()                      | R.adjoint()                       |
| Non-conjugate transpose      | R.T()                      | R.transpose()                     |
| Conjugate of array values    | conjg(R)                   |                                   |
| Flip left-right              | flip(R, 0)                 | R.rowwise().reverse()             |
| Flip up-down                 | flip(R, 1)                 | R.colwise().reverse()             |
| Repeat matrix                | tile(R, i, j)              | R.replicate(i, j)                 |
| Swap axis                    | reorder(R, 1, 0)           |                                   |
| Bind columns                 | join(1, A1, A2)            |                                   |
| Bind rows                    | join(0, A1, A2)            |                                   |
| Shift                        | shift(A, 1)                |                                   |
| Rotate                       | rotate(A, 3)               | R.transpose().colwise().reverse() |

### Matrix Math
|                              | ArrayFire (C++)              | Eigen (C++)                                                                   |
| :--------------------------- | :------------------------:   | :----------------------------:                                                |
| Triangular,  upper           | lower(R)                     |                                                                               |
| Triangular,  lower           | upper(R)                     |                                                                               |
| Rank                         | rank(R)                      |                                                                               |
| Determinant                  | det\<float\>(R)                | R.determinant()                                                               |
| Euclid  norm                 | norm(R)                      | R.squaredNorm()                                                               |
| L1 norm                      | norm(R, AF_NORM_VECTOR_1)    | R.lpNorm\<1\>()                                                                 |
| L2 norm                      | norm(R, AF_NORM_VECTOR_2)    | R.lpNorm\<2\>()                                                                 |
| L4 norm                      | norm(R, AF_NORM_VECTOR_P_4)  | R.lpNorm\<4\>()                                                                 |
| L.inf norm                   | norm(R, AF_NORM_VECTOR_INF)  | R.lpNorm\<Infinity\>()                                                          |
| Addition                     | A1+A2                        | R.array() += s                                                                |
|                              |                              | R = P.array() + s.array() s is scale                                          |
|                              |                              | R  = P + Q R += Q                                                             |
| Subtraction                  | A1-A2                        | R.array() -= s                                                                |
|                              |                              | R = P.array() - s.array() sis scale                                           |
|                              |                              | R  = P - Q R -= Q                                                             |
| Multiplication               | A1.A2                        | R = P.cwiseProduct(Q)                                                         |
|                              |                              | R = P.array() . s.array()                                                     |
| Division                     | R=R/Q                        | R = P.cwiseQuotient(Q)                                                        |
|                              |                              | R = P.array() / Q.array();                                                    |
|                              |                              | R /= s                                                                        |
| Matrix Multiplication        | matmul(A1, A2)               | P.Q R .= Q                                                                    |
|                              |                              | R  = P.s R  = s.P R .= s                                                      |
|                              |                              | y  = M.x  a  = b.M a .= M                                                     |
| Dot Product(vector)          | dot(x1, x2)                  | x.dot(y)                                                                      |
| Solve equation               | solve(A, B)                  |                                                                               |
| Inverse                      | inverse(A)                   | A.inverse()                                                                   |
| LU                           | lu(out,  pivot,  A)          | .lu()   -> .matrixL() and .matrixU()                                          |
| Cholesky factorization       | cholesky(R, A, true)         |                                                                               |
| QR                           | qr(Q, R, tau, A)             | R.qr()   -> .matrixQ() and .matrixR()                                         |
| Singular values              | svd(U, s, Vt, A)             | BDCSVD JacobiSVD\<Eigen::MatrixXf\> svd(A,  ComputeThinU . ComputeThinV )       |
|                              |                              | auto svd = A.bdcSvd(ComputeThinU . ComputeThinV);                             |
|                              |                              | svd.matrixU() . svd.singularValues().asDiagonal() . svd.matrixV().transpose() |

### Statistics
|                                | ArrayFire (C++)      | Eigen (C++)              |
| :---------------------------   | :------------------: | :----------------------: |
| Sums of columns                | sum(A1)              | R.colwise().sum()        |
| Sums of rows                   | sum(A1, 1)           | R.rowwise().sum()        |
| Sums of all elem.              | sum\<float\>(A1)       | R.sum()                  |
| Products of columns            | product(A1)          | R.colwise().prod()       |
| Products of rows               | product(A1, 1)       | R.rowwise().prod()       |
| Products of all elem.          | product\<float\>(A1)   | R.prod()                 |
| Averages of columns            | mean(A1)             | R.colwise().mean()       |
| Averages of rows               | mean(A1, 1)          | R.rowwise().mean()       |
| Averages of all elem.          | mean\<float\>(A1)      | R.mean()                 |
| Maximum of columns             | max(A1)              | R.colwise().maxCoeff()   |
| Maximum of rows                | max(A1, 1)           | R.rowwise().maxCoeff()   |
| Maximum of all elem.           | max\<float\>(A1)       | R.maxCoeff()             |
| Minimum of columns             | min(A1)              | R.colwise().minCoeff()   |
| Minimum of rows                | min(A1, 1)           | R.rowwise().minCoeff()   |
| Minimum of all elem.           | min\<float\>(A1)       | R.minCoeff()             |
| Variance  of columns           | var(A1, 1, 0)        |                          |
| Variance  of rows              | var(A1, 1, 1)        |                          |
| Variance s of all elem.        | var\<float\>(A1)       |                          |
| Stand. deviations of columns   | stdev(A1)            |                          |
| Stand. deviations of rows      | stdev(A1, 1)         |                          |
| Stand. deviations of all elem. | stdev\<float\>(A1)     |                          |

### Basic Math
|                              | ArrayFire (C++)            | Eigen (C++)                    |
| :--------------------------- | :------------------------: | :----------------------------: |
| Round                        | round(A)                   | R.array().round()              |
| Round down                   | floor(A)                   | R.array().floor()              |
| Round up                     | ceil(A)                    | R.array().ceil()               |
| Exponential                  | exp(A1)                    | R.array().exp()                |
| Element-wise power           | pow(A, 2)                  | R.array().pow(2)               |
|                              | pow(A, 4)                  | R.array().pow(2)               |

### Signal Processing
|                              | ArrayFire (C++)            | Eigen (C++)                    |
| :--------------------------- | :------------------------: | :----------------------------: |
| FFT1D (each column)          | fft(A)                     |                                |
| FFT1D (each row)             | fft(A.T).T                 |                                |
