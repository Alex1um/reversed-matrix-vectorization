//
// Created by alex1um on 25.01.23.
//

#ifndef LAB4_VECTORIZE__AUTO_H_
#define LAB4_VECTORIZE__AUTO_H_

#include "shared.h"

namespace autoopt {

template<typename M>
void MatrixMultiply(M &out, M &in1, M &in2) {
  for (size_t i = 0; i < N; i++) {
    for (size_t k = 0; k < N; k++) {
      float pt = in1[i][k];
      for (size_t j = 0; j < N; j++) {
        out[i][j] += pt * in2[k][j];
      }
    }
  }
}

template<typename M>
void MatrixSum(M &out, M &in1, M &in2) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      out[i][j] = in1[i][j] + in2[i][j];
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
    for (size_t j = 0; j < N; j++) {
      out[i][j] = in1[i][j] - in2[i][j];
    }
  }
}

template<typename T, typename M>
void MatrixMultScalar(M &in_out, T sc) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      in_out[i][j] *= sc;
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
M MatrixReverseAutoOptTime(M &A, int m) {
  M B = MatrixCreate<T>();
//  MatrixCopy(B, A);

  M BA = MatrixCreate<T>();

  M I = MatrixCreate<T>();
  MatrixFillOne(I);

  M R = MatrixCreate<T>();

  M Rpow = MatrixCreate<T>();
  MatrixFillOne(Rpow);

  M tmp = MatrixCreate<T>();

  M Arev = MatrixCreate<T>();

  auto start = c::now();
  autoopt::MatrixTranspose(B, A);
  T coef = autoopt::MatrixGetMaxMult<T, M>(A);

  autoopt::MatrixMultScalar(B, 1 / coef);

  autoopt::MatrixMultiply(BA, B, A);

  autoopt::MatrixMinus(R, I, BA);

  for (m -= 1; m > 0; m--) {
    MatrixCopy(tmp, Rpow);
    MatrixFillZero(Rpow);
    autoopt::MatrixMultiply(Rpow, tmp, R);
    autoopt::MatrixSum(I, I, Rpow);
  }
  autoopt::MatrixMultiply(Arev, I, B);
  cout << "Execution time: " << (chrono::duration_cast<t>(c::now() - start)).count() << endl;

  return Arev;
}

#endif //LAB4_VECTORIZE__AUTO_H_
