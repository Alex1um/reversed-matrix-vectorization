cmake_minimum_required(VERSION 3.24)
project(lab4_vectorize)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-mavx2 -mfma)
add_compile_options( -O3 -ffast-math )

#target_compile_features(lab4_vectorize avx)
add_executable(lab4_vectorize main.cpp shared.h default.h auto.h)
#target_link_libraries(lab4_vectorize ${BLAS_LIBRARIES})
