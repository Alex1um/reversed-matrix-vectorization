#include <iostream>

#include "blas.h"
#include "shared.h"
#include "default.h"
#include "auto.h"
#include "simd.h"

template<typename M>
void MatrixFill(M &in) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
//      if (i == j)
        in[i][j] = i * N + j;
//      in[i][j] = 1;
    }
  }
}

template<typename T, typename M>
void test() {

  M input = MatrixCreate<T>();
  MatrixFill(input);

  MatrixPrint(input, 4);
  M out;
//  out = MatrixReverseUnoptTime<T, M, nanoseconds>(input, 1);
//  MatrixPrint(out, 4);

  out = MatrixReverseAutoOptTime<T, M, nanoseconds>(input, 2);
  MatrixPrint(out, 4);

//  out = MatrixReverseSimd<T, M, nanoseconds>(input, 1);
//  MatrixPrint(out, 4);

  out = MatrixReverseCBlas<T, M, nanoseconds>(input, 2);
  MatrixPrint(out, 4);
}

int main() {
  test<float, float (*)[N]>();
  return 0;
}
