#include <iostream>

#include "blas.h"
#include "shared.h"
#include "default.h"
#include "auto.h"
#include "simd.h"

template<typename M>
void MatrixFill(M &in) {
//  srand(time(0));
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
//      if (i == j)
        in[i][j] = i * N + j;
//      in[i][j] = 1;
//        in[i][j] = rand();
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
