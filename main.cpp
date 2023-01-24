#include <iostream>

#include "shared.h"
#include "default.h"
#include "auto.h"

template<typename M>
void MatrixFill(M &in) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      if (i == j)
        in[i][j] = i * N + j;
//      in[i][j] = 1;
    }
  }
}

template<typename T, typename M>
void test() {
  M input = MatrixCreate<T>();
  MatrixFill(input);

  MatrixPrint(input, N);

  M out = MatrixReverseUnoptTime<T, M, nanoseconds>(input, 50);
  MatrixPrint(out, N);

  out = MatrixReverseAutoOptTime<T, M, nanoseconds>(input, 50);
  MatrixPrint(out, N);
}

int main() {
  test<float, float (*)[N]>();
  return 0;
}
