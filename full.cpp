//
// Created by alex1um on 26.01.23.
//
#include <iostream>
#include <chrono>
#include <cblas.h>
#include <x86intrin.h>

using namespace std;
using namespace std::chrono;

#define N 2048

template<typename T>
T (*(MatrixCreate()))[N] {
  auto temp = new T[N][N];
  return temp;
}

template<typename M>
void MatrixPrint(M const &in, size_t n) {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      cout << in[i][j] << ' ';
    }
    cout << endl;
  }
  cout << endl;
}

template<typename M>
void MatrixCopy(M &out, M const &in) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      out[i][j] = in[i][j];
    }
  }
}

template<typename M>
void MatrixFillOne(M &in) {
  for (size_t i = 0; i < N; i++) {
    in[i][i] = 1;
  }
}

template<typename M>
void MatrixFillZero(M &in_out) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      in_out[i][j] = 0;
    }
  }
}

namespace unopt {

template<typename M>
void MatrixMultiply(M &out, M &in1, M &in2) {
  for (size_t i = 0; i < N; i++) {
    for (size_t k = 0; k < N; k++) {
      float pt = in1[i][k];
#pragma clang loop vectorize(disable)
      for (size_t j = 0; j < N; j++) {
        out[i][j] += pt * in2[k][j];
      }
    }
  }
}

template<typename M>
void MatrixSum(M &out, M &in1, M &in2) {
  for (size_t i = 0; i < N; i++) {
#pragma clang loop vectorize(disable)
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
#pragma clang loop vectorize(disable)
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
#pragma clang loop vectorize(disable)
    for (size_t j = 0; j < N; j++) {
      out[i][j] = in1[i][j] - in2[i][j];
    }
  }
}

template<typename T, typename M>
void MatrixMultScalar(M &in_out, T sc) {
  for (size_t i = 0; i < N; i++) {
#pragma clang loop vectorize(disable)
    for (size_t j = 0; j < N; j++) {
      in_out[i][j] *= sc;
    }
  }
}

template<typename M>
void MatrixTranspose(M &out, M &in) {
  for (size_t i = 0; i < N; i++) {
#pragma clang loop vectorize(disable)
    for (size_t j = 0; j < N; j++) {
      out[i][j] = in[j][i];
    }
  }
}
}

template<typename T, typename M>
M MatrixReverseUnopt(M &A, int m) {
  M B = MatrixCreate<T>();
  unopt::MatrixTranspose(B, A);
  T coef = unopt::MatrixGetMaxMult<T, M>(A);
  unopt::MatrixMultScalar(B, 1 / coef);
  M I = MatrixCreate<T>();
  MatrixFillOne(I);
  M BA = MatrixCreate<T>();
  unopt::MatrixMultiply(BA, B, A);
  M R = MatrixCreate<T>();
  unopt::MatrixMinus(R, I, BA);
  M Rpow = MatrixCreate<T>();
  MatrixFillOne(Rpow);
  M tmp = MatrixCreate<T>();
  for (m -= 1; m > 0; m--) {
    MatrixCopy(tmp, Rpow);
    unopt::MatrixMultiply(Rpow, tmp, R);
    unopt::MatrixSum(I, I, Rpow);
  }
  M Arev = MatrixCreate<T>();
  unopt::MatrixMultiply(Arev, I, B);
  return Arev;
}

template<typename T, typename M, typename t, typename c = chrono::steady_clock>
M MatrixReverseUnoptTime(M &A, int m) {
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
  unopt::MatrixTranspose(B, A);
  T coef = unopt::MatrixGetMaxMult<T, M>(A);
  unopt::MatrixMultScalar(B, 1 / coef);
  unopt::MatrixMultiply(BA, B, A);
  unopt::MatrixMinus(R, I, BA);
  for (m -= 1; m > 0; m--) {
    MatrixCopy(tmp, Rpow);
    MatrixFillZero(Rpow);
    unopt::MatrixMultiply(Rpow, tmp, R);
    unopt::MatrixSum(I, I, Rpow);
  }
  unopt::MatrixMultiply(Arev, I, B);
  cout << "Execution time: " << (chrono::duration_cast<t>(c::now() - start)).count() << endl;
  return Arev;
}

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

namespace intropt {

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
M MatrixReverseIntrisics(M &A, int m) {
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
  intropt::MatrixTranspose(B, A);
  T coef = intropt::MatrixGetMaxMult<T, M>(A);
  intropt::MatrixMultScalar(B, 1 / coef);
  intropt::MatrixMultiply(BA, B, A);
  intropt::MatrixMinus(R, I, BA);
  for (m -= 1; m > 0; m--) {
    MatrixCopy(tmp, Rpow);
    MatrixFillZero(Rpow);
    intropt::MatrixMultiply(Rpow, tmp, R);
    intropt::MatrixSum(I, I, Rpow);
  }
  intropt::MatrixMultiply(Arev, I, B);
  cout << "Execution time: " << (chrono::duration_cast<t>(c::now() - start)).count() << endl;
  return Arev;
}

namespace blas {

template<typename T = float, typename M>
T MatrixGetMaxMult(M &in) {
  float max_row = 1e-38, max_col = 1e-38;
  for (size_t i = 0; i < N; i++) {
    float row = cblas_sasum(N, in[i], 1);
    if (max_row < row) max_row = row;

    float col = cblas_sasum(N, &in[0][i], N);
    if (max_col < col) max_col = col;
  }
  return max_row * max_col;
}

template<typename M>
void MatrixSum(M &out, M &in1, M &in2) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      out[i][j] = in1[i][j] + in2[i][j];
    }
  }
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
  M Rpow = MatrixCreate<T>();
  MatrixFillOne(Rpow);
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
      (float *) A,
      N,
      (float *) A,
      N,
      1.0,
      (float *) R,
      N
  );
  for (m -= 1; m > 0; m--) {
    MatrixCopy(tmp, Rpow);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N, N, N,
        1.0,
        (float *) tmp,
        N,
        (float *) R,
        N,
        0.0,
        (float *) Rpow,
        N
    );
    blas::MatrixSum(Rsum, Rsum, Rpow);
  }
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,N, N, N,
      1 / coef,
      (float *) Rsum,
      N,
      (float *) A,
      N,
      0,
      (float *) Arev,
      N
  );
  cout << "Execution time: " << (chrono::duration_cast<t>(c::now() - start)).count() << endl;
  return Arev;
}

template<typename M>
void MatrixFill(M &in) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      in[i][j] = i * N + j;
    }
  }
}

template<typename T, typename M, typename A>
void test() {
  M input = MatrixCreate<T>();
  MatrixFill(input);
  int m = 10;
  MatrixPrint(input, 4);
  M out;
  out = MatrixReverseUnoptTime<T, M, A>(input, m);
  MatrixPrint(out, 4);
  out = MatrixReverseAutoOptTime<T, M, A>(input, m);
  MatrixPrint(out, 4);
  out = MatrixReverseIntrisics<T, M, A>(input, m);
  MatrixPrint(out, 4);
  out = MatrixReverseCBlas<T, M, A>(input, m);
  MatrixPrint(out, 4);
}

int main() {
  test<float, float (*)[N], microseconds>();
  return 0;
}
