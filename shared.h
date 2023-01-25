//
// Created by alex1um on 24.01.23.
//

#ifndef LAB4_VECTORIZE__SHARED_H_
#define LAB4_VECTORIZE__SHARED_H_

#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

//#define N 2048
#define N 64

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

#endif //LAB4_VECTORIZE__SHARED_H_
