//
// Created by alex1um on 25.01.23.
//

#ifndef LAB4_VECTORIZE__SIMD_H_
#define LAB4_VECTORIZE__SIMD_H_

#include "shared.h"
#include <x86intrin.h>
#include <algorithm>

namespace simdopt {

template<typename M = float **>
void MatrixMultiply(M &out, M &in1, M &in2) {
  for (size_t i = 0; i < N; i++) {
    for (size_t k = 0; k < N; k++) {
      __m256 pt = _mm256_set1_ps(in1[i][k]);
      for (size_t j = 0; j < N; j += (sizeof(__m256) / sizeof(float))) {
        __m256 c = _mm256_loadu_ps(&in2[k][j]);
        __m256 mul = _mm256_mul_ps(c, pt);
        __m256 res = _mm256_loadu_ps(&out[i][j]);
        __m256 add = _mm256_add_ps(mul, res);
        _mm256_storeu_ps(&out[i][j], add);
      }
    }
  }
}


template<typename M>
void MatrixSum(M &out, M &in1, M &in2) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j += (sizeof(__m256) / sizeof(float))) {
      __m256 p1 = _mm256_loadu_ps(&in1[i][j]);
      __m256 p2 = _mm256_loadu_ps(&in2[i][j]);
      __m256 s = _mm256_add_ps(p1, p2);
      _mm256_storeu_ps(&out[i][j], s);
    }
  }
}

template<typename T = float, typename M>
T MatrixGetMaxMult(M &in) {
  T max_row = 1e-38, max_col = 1e-38;
  for (size_t i = 0; i < N; i++) {
    T row = 0, col = 0;
    for (size_t j = 0; j < N; j++) {
      row += abs(in[i][j]);
      col += abs(in[j][i]);
    }
    if (max_row < row) max_row = row;
    if (max_col < col) max_col = col;
  }
  return max_row * max_col;
}

template<typename M>
void MatrixMinus(M &out, M &in1, M &in2) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j += (sizeof(__m256) / sizeof(float))) {
      __m256 p1 = _mm256_loadu_ps(&in1[i][j]);
      __m256 p2 = _mm256_loadu_ps(&in2[i][j]);
      __m256 s = _mm256_sub_ps(p1, p2);
      _mm256_storeu_ps(&out[i][j], s);
//      out[i][j] = in1[i][j] - in2[i][j];
    }
  }
}

template<typename T = float, typename M>
void MatrixMultScalar(M &in_out, T sc) {
  __m256 scm = _mm256_set1_ps(sc);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j += (sizeof(__m256) / sizeof(float))) {
      __m256 p1 = _mm256_loadu_ps(&in_out[i][j]);
      __m256 res = _mm256_mul_ps(p1, scm);
      _mm256_storeu_ps(&in_out[i][j], res);
//      in_out[i][j] *= sc;
    }
  }
}

template<typename M>
void MatrixTranspose(M &out, M &in) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      out[i][j] = in[j][i];
    }
  }
}

}

template<typename T, typename M, typename t, typename c = chrono::steady_clock>
M MatrixReverseSimd(M &A, int m) {
  M B = MatrixCreate<T>();

  M BA = MatrixCreate<T>();

  M I = MatrixCreate<T>();
  MatrixFillOne(I);

  M R = MatrixCreate<T>();

  M Rpow = MatrixCreate<T>();
  MatrixFillOne(Rpow);

  M tmp = MatrixCreate<T>();

  M Arev = MatrixCreate<T>();

  auto start = c::now();
  simdopt::MatrixTranspose(B, A);
  T coef = simdopt::MatrixGetMaxMult<T, M>(A);
  simdopt::MatrixMultScalar(B, 1 / coef);

  simdopt::MatrixMultiply(BA, B, A);

  simdopt::MatrixMinus(R, I, BA);

  for (m -= 1; m > 0; m--) {
    MatrixCopy(tmp, Rpow);
    MatrixFillZero(Rpow);
    simdopt::MatrixMultiply(Rpow, tmp, R);
    simdopt::MatrixSum(I, I, Rpow);
  }
  simdopt::MatrixMultiply(Arev, I, B);
  cout << "Execution time: " << (chrono::duration_cast<t>(c::now() - start)).count() << endl;

  return Arev;
}

#endif //LAB4_VECTORIZE__SIMD_H_
