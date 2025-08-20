#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/AutoreleasePoolGuard.hpp>
#include <Metal/MetalBuffer.hpp>
#include <Metal/MetalContext.hpp>
#include <iostream>
#include <utils/util.hpp>

#define NROWS 3
#define NCOLS 3

void host_matrix_multiply(float *matA, float *matB, float *matC, int nrows,
                          int ncols) {
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            float sum = 0;
            for (int k = 0; k < nrows; ++k) {
                sum += matA[i * nrows + k] * matB[k * ncols + j];
            }
            matC[i * nrows + j] = sum;
        }
    }
}

int main() {
    AutoreleasePoolGuard guard;
    uint nrows = NROWS;

    std::unique_ptr<float[]> matA = std::make_unique<float[]>(NROWS * NCOLS);
    auto matB = std::make_unique<float[]>(NROWS * NCOLS);
    auto matC = std::make_unique<float[]>(NROWS * NCOLS);
    auto matH = std::make_unique<float[]>(NROWS * NCOLS);

    populate_matrix(matA.get(), NROWS, NCOLS);
    populate_matrix(matB.get(), NROWS, NCOLS);

    MetalContext multiplier("build/matmul_kernel.metallib",
                            "device_matrix_multiply");

    MetalBuffer bufA(multiplier, sizeof(float) * NROWS * NCOLS);
    MetalBuffer bufB(multiplier, sizeof(float) * NROWS * NCOLS);
    MetalBuffer bufC(multiplier, sizeof(float) * NROWS * NCOLS);
    MetalBuffer bufWidth(multiplier, sizeof(uint));

    bufA.fillBuffer(matA.get(), sizeof(float) * NROWS * NCOLS);
    bufB.fillBuffer(matB.get(), sizeof(float) * NROWS * NCOLS);
    bufWidth.fillBuffer(&nrows, sizeof(uint));

    multiplier.setBuffer(bufA, 0, 0);
    multiplier.setBuffer(bufB, 0, 1);
    multiplier.setBuffer(bufC, 0, 2);
    multiplier.setBuffer(bufWidth, 0, 3);

    MetalDim gridDim(NROWS, NCOLS, 1);
    MetalDim blockDim(NROWS, NCOLS, 1);

    multiplier.runKernel(gridDim, blockDim);

    std::memcpy(matC.get(), bufC.contents(), sizeof(float) * NROWS * NCOLS);

    host_matrix_multiply(matA.get(), matB.get(), matH.get(), NROWS, NCOLS);

    if (compare_matrices(matC.get(), matH.get(), NROWS, NCOLS)) {
        std::cout << "Matrix multiplication matches" << std::endl;
    } else {
        std::cout << "Matrix multiplication "
                     "does not match"
                  << std::endl;
    }

    return 0;
}
