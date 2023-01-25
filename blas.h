//
// Created by alex1um on 26.01.23.
//

#ifndef LAB4_VECTORIZE__BLAS_H_
#define LAB4_VECTORIZE__BLAS_H_

#include <cblas.h>
#include "shared.h"

/*
 * Процедура xGEMM вычисляет следующее выражение: С = αAB + βC
 * Order - определяет порядок следования элементов:
                        CblasRowMajor - матрицы хранятся по строкам (стандартно в C),
                        CblasColMajor - матрицы хранятся по столбцам;
            TransA, TransB - определяет предварительные операции над матрицами A и B:
                        CblasNoTrans - ничего не делать,
                        CblasTrans - транспонировать,
                        CblasConjTrans - вычислить сопряженную матрицу;
            M, N, K          - размеры матриц;
            alpha, beta       - коэффициенты;
lda, ldb, ldc - число элементов в ведущей размерности матрицы (строке или столбце). Для массивов языка Си - число элементов в строке:
            lda = K
            ldb = N
            ldc = N
void cblas_sgemm(const enum CBLAS_ORDER Order,
                        const enum CBLAS_TRANSPOSE TransA,
                        const enum CBLAS_TRANSPOSE TransB,
                        const int M,
                        const int n,
                        const int k,
                        const float alpha,
                        const float *A,
                        const int lda,
                        const float *B,
                        const int ldb,
                        const float beta,
                        float *C,
                        const int ldc);
*/

namespace blas {


template<typename T = float, typename M>
T MatrixGetMaxMult(M &in) {
  float max_row = 1e-38, max_col = 1e-38;
  for (size_t i = 0; i < N; i++) {
    float row = cblas_sasum(N, in[i], 1);
    if (max_row < row) max_row = row;

    float col = 0;
    for (size_t j = 0; j < N; j++) {
      col += abs(in[j][i]);
    }
    if (max_col < col) max_col = col;
  }
  return max_row * max_col;
}

}

template<typename T, typename M, typename t, typename c = chrono::steady_clock>
M MatrixReverseCBlas(M &A, int m) {
  M I = MatrixCreate<T>();
  MatrixFillOne(I);

  M R = MatrixCreate<T>();
  MatrixFillOne(R);

  M Rsum = MatrixCreate<T>();
  MatrixFillOne(Rsum);

  M tmp = MatrixCreate<T>();

  M Arev = MatrixCreate<T>();

  auto start = c::now();
  T coef = blas::MatrixGetMaxMult<T, M>(A);

  cblas_sgemm(
      CblasRowMajor,
      CblasTrans,
      CblasNoTrans,
      N, N, N,
      -1 / coef,
      (float *)A,
      N,
      (float *)A,
      N,
      1.0,
      (float *)R,
      N
      );

  for (m -= 1; m > 0; m--) {
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N, N, N,
        1.0,
        (float *)R,
        N,
        (float *)I,
        N,
        1.0,
        (float *)Rsum,
        N
    );
  }
  cblas_sgemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasTrans,
      N, N, N,
      1 / coef,
      (float *)Rsum,
      N,
      (float *)A,
      N,
      0,
      (float *)Arev,
      N
  );

  cout << "Execution time: " << (chrono::duration_cast<t>(c::now() - start)).count() << endl;

  return Arev;
}

#endif //LAB4_VECTORIZE__BLAS_H_
