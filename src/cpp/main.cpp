#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// #include "Foundation/NSString.hpp"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <util.h>

#define NROWS 32
#define NCOLS 32

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

class GPUMatrixMultiplier {
  private:
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::CommandQueue> m_queue;
    NS::SharedPtr<MTL::Library> m_lib;
    NS::SharedPtr<MTL::Function> m_fn;
    NS::SharedPtr<MTL::ComputePipelineState> m_pipeline;

  public:
    GPUMatrixMultiplier(const char *lib, const char *func) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        m_queue = NS::TransferPtr(m_device->newCommandQueue());

        NS::Error *error = nullptr;
        auto *library = NS::String::string(lib, NS::UTF8StringEncoding);
        m_lib = NS::TransferPtr(m_device->newLibrary(library, &error));

        if (!m_lib) {
            std::cerr << "Failed to create library "
                      << error->localizedDescription()->utf8String()
                      << std::endl;
            pool->release();
            throw std::runtime_error("Failed to create Metal library");
        }

        auto *function = NS::String::string(func, NS::UTF8StringEncoding);
        m_fn = NS::TransferPtr(m_lib->newFunction(function));

        if (!m_fn) {
            pool->release();
            throw std::runtime_error("Failed to find function in library");
        }

        m_pipeline = NS::TransferPtr(
            m_device->newComputePipelineState(m_fn.get(), &error));

        pool->release();
    }

    ~GPUMatrixMultiplier() = default;

    void multiplyMatrixGPU(float *matA, float *matB, float *matC) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

        size_t matrixSizeBytes = NROWS * NCOLS * sizeof(float);

        uint nrows = NROWS;

        auto bufA = NS::TransferPtr(m_device->newBuffer(
            matA, matrixSizeBytes, MTL::ResourceStorageModeManaged));
        auto bufB = NS::TransferPtr(m_device->newBuffer(
            matA, matrixSizeBytes, MTL::ResourceStorageModeManaged));
        auto bufC = NS::TransferPtr(m_device->newBuffer(
            matA, matrixSizeBytes, MTL::ResourceStorageModeManaged));
        auto bufWidth = NS::TransferPtr(m_device->newBuffer(
            &nrows, sizeof(uint), MTL::ResourceStorageModeManaged));

        auto commandBuffer = NS::TransferPtr(m_queue->commandBuffer());

        auto encoder = NS::TransferPtr(commandBuffer->computeCommandEncoder());

        encoder->setComputePipelineState(m_pipeline.get());
        encoder->setBuffer(bufA.get(), 0, 0);
        encoder->setBuffer(bufB.get(), 0, 1);
        encoder->setBuffer(bufC.get(), 0, 2);
        encoder->setBuffer(bufWidth.get(), 0, 3);

        MTL::Size grid(NROWS, NCOLS, 1);
        MTL::Size threadsPerThreadgroup(NROWS, NCOLS,
                                        1); // Better threadgroup size

        encoder->dispatchThreads(grid, threadsPerThreadgroup);
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        // Check for command buffer errors
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            std::cerr << "Command buffer execution failed" << std::endl;
            if (commandBuffer->error()) {
                std::cerr << "Error: "
                          << commandBuffer->error()
                                 ->localizedDescription()
                                 ->utf8String()
                          << std::endl;
            }
        }

        memcpy(matC, bufC.get()->contents(), sizeof(float) * NROWS * NCOLS);
    }
};

int main() {
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    float *matA = (float *)malloc(sizeof(float) * NROWS * NCOLS);
    float *matB = (float *)malloc(sizeof(float) * NROWS * NCOLS);
    float *matC = (float *)malloc(sizeof(float) * NROWS * NCOLS);
    float *matH = (float *)malloc(sizeof(float) * NROWS * NCOLS);

    populate_matrix(matA, NROWS, NCOLS);
    populate_matrix(matB, NROWS, NCOLS);

    GPUMatrixMultiplier multiplier("build/matmul_kernel.metallib",
                                   "device_matrix_multiply");

    multiplier.multiplyMatrixGPU(matA, matB, matC);

    host_matrix_multiply(matA, matB, matH, NROWS, NCOLS);

    if (compare_matrices(matC, matH, NROWS, NCOLS)) {
        std::cout << "Matrix multiplication matches" << std::endl;
    } else {
        std::cout << "Matrix multiplication does not match" << std::endl;
    }

    return 0;
}
