This document enumerates arrayfire API equivalent in Matlab, Python (numpy, scipy).
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
|                              | ArrayFire (C++)            | Matlab                         | Python (numpy, scipy)                      |
| :--------------------------- | :------------------------: | :----------------------------: | :----------------------------------------: |
| Identity matrix              | identity(rows, cols)       | eye(rows, cols)                | np.identity(rows, 'float32')               |
| Array filled with constant n | constant(n, rows, cols)    | ones(rows, cols).n             | np.ones((rows, cols), 'float32').n         |
| Random filled array          | randu(rows, cols)          | rand(rows, cols)               | rand(rows, cols)                           |
| Diagonal vector to matrix    | diag(A)                    | diag(A)                        | np.diag(A)                                 |

### Fetch Array Info
|                         | AarrayFire (C++)               | Matlab          | Python (numpy, scipy)     |
| :---------------------- | :----------------------------: | :-------------: | :-----------------------: |
|                         | array A(rows, cols, f64)       |                 |                           |
| Vector size             | A.elements()                   | length(A)       | A.size                    |
| Number of dimensions    | A.ndims()                      | ndims(A)        | len(x.shape)              |
| shape of matrix         | A.dims()                       | size（A）       | A.shape                   |
| number of rows          | A.dims(0)                      | size(A, 1)      | A.shape.[0]               |
| number of columns       | A.dims(1)                      | size(A, 2)      | A.shape.[1]               |
| number of elements      | A.elements()                   | numel(A)        | A.size                    |


### Indexing
|                               | ArrayFire (C++)                           | Matlab                              | Python (numpy, scipy)           |
| :---------------------------- | :----------------------------------------:| :----------------------------------:| :------------------------------:|
| Sequences	                    | af::seq(low,high,step)	                | (low:step:high)'->columnvector	  |                                 |
| **Vector**                    |                                           |                                     |                                 |
| i+1 element                   | A(i)                                      | A(i+1)                              | A[i]                            |
| last element                  | A(end)                                    | A(end)                              |                                 |
|                               | A(-1)                                     | A(end)                              | A[-1]                           |
| 1 to i elements               | A(seq(i))                                 | A(1:i)                              | A[:i]                           |
| i+1 to END elements           | A(seq(i, end))                            | A(i+1:end)                          | A[i:]                           |
| i1+1 to i2+1 elements         | A(seq(i1, i2))                            | A(i1+1:i2+1)                        | A[i1:i2+1]                      |
| i1+1 to i2+1 elements by step | A(seq(i1, i2, step))                      | A(i1+1:step:i2+1)                   | A[i1:i2+1:step]                 |
| **Matrix**                    |                                           |                                     |                                 |
| i+1 element                   | A(i)                                      | A(i+1)                              | A[i]                            |
| last element                  | A(end)                                    | A(end)                              |                                 |
|                               | A(-1)                                     | A(end)                              | A[-1]                           |
| Element(i+1, j+1)             | A(i, j)                                   | A(i+1, j+1)                         | A[i, j]                         |
| i+1 row                       | A(i, span)                                | A(i+1, :)                           | A[i, :]                         |
| i+1 column                    | A(span, i)                                | A(:, i+1)                           | A[:, i]                         |
|                               | A(seq(i, end), seq(j, end))               | A(i+1:end, j+1:end)                 |                                 |
|                               | A(seq(i, end), span)                      | A(i+1:end, :)                       | A[i:, :]                        |
|                               | A(seq(i1, i2), seq(j1, j2))               | A(i1+1:12+1, j1+1:j2+1)             | A[i1:i2+1, j1:j2+1]             |
|                               | A(seq(i1, i2, step1), seq(j1, j2, step2)) | A(i1+1:step1:12+1, j1+1:step2:j2+1) | A[i1:i2+1:step1, j1:j2+1:step2] |
| i+1 row                       | A.row(i)                                  | A(i+1, :)                           | A[i, :]                         |
| i+1 to j+1 rows               | A.rows(i, j)                              | A(i+1:j+1, :)                       | A[i:j+1, :]                     |
| i+1 column                    | A.col(i)                                  | A(:, i+1)                           | A[:, i]                         |
| i+1 to j+1 columns            | R.cols(i, j)                              | A(:, i+1:j+1)                       | A[:, i:j+1]                     |
| i+1 slice                     | A.slice(i)                                | A(:, :, i+1)                        | A[:, :, i]                      |
| i+1 to j+1 slices             | A.slices(i, j)                            | A(:, :, i+1:j+1)                    | A[:, :, i:j+1]                  |


### Array Reshape
|                               | ArrayFire (C++)           | Matlab                      | Python (numpy, scipy)                       |
| :---------------------------- | :-----------------------: | :-------------------------: | :------------------------------------------:|
| Flatten to vector             | flat(R)                   | R(:)                        | R.flatten()                                 |
| Reshaping                     | moddims(R, rows, cols)    | reshape(R, [rows, cols])    | np.reshape(R, (rows, cols))                 |
| Conjugate Transposition       | R.H()                     | R'                          | R.conj().T                                  |
| Non-conjugate transpose       | R.T()                     | R.'                         | R.T                                         |
| Conjugate of array values     | conjg(R)                  | conj(R)                     | np.conj                                     |
| Flip left-right               | flip(R, 1)                | fliplr(R)                   | np.fliplr(R)                                |
| Flip up-down                  | flip(R, 0)                | flipud(R)                   | np.flipud(R)                                |
| Repeat matrix                 | tile(R, i, j)             | repmat(R, i, j)             | np.tile(R, (i, j))                          |
| Swap axis                     | reorder(R, 1, 0)          | permute(R, [2, 1])          | arr.transpose(2, 1, 0) arr.swapaxes(2, 1)   |
| Bind columns                  | join(1, A1, A2)           | [A1;A2] vertcat(A1, A2)     | vstack((a, b)) concatenate((a, b),  axis=0) |
| Bind rows                     | join(0, A1, A2)           | [A1A2] horzcat(A1, A2)      | hstack((a, b)) concatenate((a, b),  axis=1) |
| Shift                         | shift(A, 1)               | circshift(A, 1)             | np.roll(R, 1)                               |
| Rotate                        | rotate(A, 3)              | rot90(A)                    | np.rot90(a)                                 |


### Matrix Math
|                          | ArrayFire (C++)            | Matlab         | Python (numpy, scipy)       |
| :----------------------- | :-----------------------:  | :------------: | :-------------------------: |
| Lower Triangular    | lower(R)                   | tril(R)        | np.tril(R)                  |
| Upper Triangular    | upper(R)                   | triu(R)        | np.triu(R)                  |
| Rank                     | rank(R)                    | rank(R)        | rank(a)                     |
| Determinant              | det\<float\>(R)              | det(R)         | la.det(a)                   |
| Euclid  norm             | norm(R)                    | norm(R)        | la.norm(R)                  |
| L1 norm                  | norm(R,AF_NORM_VECTOR_1)   | norm(R,1)      | la.norm(R,1)                |
| L2 norm                  | norm(R,AF_NORM_VECTOR_2)   | norm(R,2)      | la.norm(R,2)                |
| L4 norm                  | norm(R,AF_NORM_VECTOR_P,4) | norm(R,4)      |                             |
| L_inf norm               | norm(R,AF_NORM_VECTOR_INF) | norm(R,Inf)    | np.linalg.norm(R,np.Inf)    |
| Addition                 | A1+A2                      | A1+A2          | A1+A2                       |
| Subtraction              | A1-A2                      | A1-A2          | A1-A2                       |
| Multiplication           | A1*A2                      | A1.*A2         | np.multiply(A1,A2)          |
|                          |                            |                | A1*A2                       |
| Division                 | R=R/Q                      | R = P / Q      | R=R/Q                       |
| Matrix Multiplication    | matmul(A1,A2)              | A1*A2          | np.matmul(A1,A2)            |
|                          |                            |                | np.dot(A1,A2)               |
|                          |                            |                | A1@A2                       |
| Dot Product(vector)      | dot(x1,x2)                 | dot(x1,x2)     | dot(x1,x2)                  |
| solve equation           | solve(A,B)                 | X=A\B          | la.solve(a,B)               |
| Inverse                  | inverse(A)                 | inv(A)         | la.inv(a)                   |
| LU                       | L,U,P=lu(A)                | [L,U,P]=lu(A)  | p,l,u=la.lu(a)              |
| Cholesky factorization   | cholesky(R,A,true)         | R=chol(A1)     | R=la.cholesky(A)            |
| QR                       | qr(Q,R,tau,A)              | [Q,R]=qr(A)    | Q,R=la.qr(A)                |
| Singular values          | svd(U,s,Vt,A)              | [U,S,V]=svd(A) | U,s,Vt=la.svd(A)            |

### Statistics
|                                | ArrayFire (C++)       | Matlab                   | Python (numpy, scipy)   |
| :----------------------------  | :-------------------: | :----------------------: | :---------------------: |
| Sums of columns                | sum(A1)               | sum(A1) sum(A1, 1)       | np.sum(A1, 0)           |
| Sums of rows                   | sum(A1, 1)            | sum(A1, 2)               | np.sum(A1, 1)           |
| Sums of all elem.              | sum\<float\>(A1)        | sum(A1, 'all')           | np.sum(A1)              |
| Products of columns            | product(A1)           | prod(A1) prod(A1, 1)     | np.prod(A1, 0)          |
| Products of rows               | product(A1, 1)        | prod(A1, 2)              | np.prod(A1, 1)          |
| Products of all elem.          | product\<float\>(A1)    | prod(A1, 'all')          | np.prod(a)              |
| Averages of columns            | mean(A1)              | mean(A1) mean(A1, 1)     | np.mean(A1, 0)          |
| Averages of rows               | mean(A1, 1)           | mean(A1, 2)              | np.mean(A1, 1)          |
| Averages of all elem.          | mean\<float\>(A1)       | mean(A1, 'all')          | np.mean(A1)             |
| Maximum of columns             | max(A1)               | max(A1) max(A1, 1)       | np.max(A1, 0)           |
| Maximum of rows                | max(A1, 1)            | max(A1, 2)               | np.max(A1, 1)           |
| Maximum of all elem.           | max\<float\>(A1)        | max(A1, 'all')           | np.max(A1)              |
| Minimum of columns             | min(A1)               | min(A1) min(A1, 1)       | np.min(A1, 0)           |
| Minimum of rows                | min(A1, 1)            | min(A1, 2)               | np.min(A1, 1)           |
| Minimum of all elem.           | min\<float\>(A1)        | min(A1, 'all')           | np.min(A1)              |
| Variance  of columns           | var(A1, 1, 0)         | var(A1) var(A1, 0, 1)    |                         |
| Variance  of rows              | var(A1, 1, 1)         | var(A1, 0, 2)            |                         |
| Variance s of all elem.        | var\<float\>(A1)        | var(A1, 0, 'all')        | np.var(x)               |
| Stand. deviations of columns   | stdev(A1)             | std(A1, 1, 1)            |                         |
| Stand. deviations of rows      | stdev(A1, 1)          | std(A1, 1, 2)            |                         |
| Stand. deviations of all elem. | stdev\<float\>(A1)      | std(A1, 1, 'all')        | np.std(x)               |

### Basic Math
|                      | ArrayFire (C++)    | Matlab       | Python (numpy, scipy)   |
| :------------------- | :----------------: | :----------: | :---------------------: |
| Exponential          | exp(A1)            | exp(A1)      | np.exp(A1)              |
| Element-wise power   | pow(A,2)           | A.^2         | np.power(A,2)           |
|                      | pow(A,4)           | A.^4         |                         |
| Round                | round(A)           | round(A)     | np.around(A)            |
| Round down           | floor(A)           | floor(A)     | np.floor(A)             |
| Round up             | ceil(A)            | ceil(A)      | np.ceil(A)              |

### Signal Processing
|                      | ArrayFire (C++)   | Matlab      | Python (numpy, scipy)           |
| :------------------- | :---------------: | :---------: | :-----------------------------: |
| FFT1D (each column)  | fft(A)            | fft(A)      | np.fft.fft(A,0)                 |
| FFT1D (each row)     | fft(A.T).T        | fft(A,2)    | np.fft.fft(A) np.fft.fft(A,1)   |
