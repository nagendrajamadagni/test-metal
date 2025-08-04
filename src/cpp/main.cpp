#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <util.h>

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

class AutoreleasePoolGuard {
  private:
    NS::AutoreleasePool *pool;

  public:
    AutoreleasePoolGuard() : pool(NS::AutoreleasePool::alloc()->init()) {}
    ~AutoreleasePoolGuard() {
        if (pool) {
            pool->release();
        }
    }
    // Prevent copying of autorelease pools
    AutoreleasePoolGuard(const AutoreleasePoolGuard &) = delete;
    AutoreleasePoolGuard &operator=(const AutoreleasePoolGuard &) = delete;
};

class GPUMatrixMultiplier {
  private:
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::CommandQueue> m_queue;
    NS::SharedPtr<MTL::Library> m_lib;
    NS::SharedPtr<MTL::Function> m_fn;
    NS::SharedPtr<MTL::ComputePipelineState> m_pipeline;
    MTL::CommandBuffer *m_command_buffer;
    MTL::ComputeCommandEncoder *m_encoder;

  public:
    GPUMatrixMultiplier(const char *lib, const char *func) {
        m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        m_queue = NS::TransferPtr(m_device->newCommandQueue());

        NS::Error *error = nullptr;
        auto *library = NS::String::string(lib, NS::UTF8StringEncoding);
        m_lib = NS::TransferPtr(m_device->newLibrary(library, &error));

        if (!m_lib) {
            std::cerr << "Failed to create library "
                      << error->localizedDescription()->utf8String()
                      << std::endl;
            throw std::runtime_error("Failed to create Metal library");
        }

        auto *function = NS::String::string(func, NS::UTF8StringEncoding);
        m_fn = NS::TransferPtr(m_lib->newFunction(function));

        if (!m_fn) {
            throw std::runtime_error("Failed to find function in library");
        }

        m_pipeline = NS::TransferPtr(
            m_device->newComputePipelineState(m_fn.get(), &error));
    }

    ~GPUMatrixMultiplier() = default;

    void multiplyMatrixGPU(float *matA, float *matB, float *matC) {

        size_t matrixSizeBytes = NROWS * NCOLS * sizeof(float);

        uint nrows = NROWS;

        auto bufA = NS::TransferPtr(m_device->newBuffer(
            matA, matrixSizeBytes, MTL::ResourceStorageModeManaged));
        auto bufB = NS::TransferPtr(m_device->newBuffer(
            matB, matrixSizeBytes, MTL::ResourceStorageModeManaged));
        auto bufC = NS::TransferPtr(m_device->newBuffer(
            matC, matrixSizeBytes, MTL::ResourceStorageModeManaged));
        auto bufWidth = NS::TransferPtr(m_device->newBuffer(
            &nrows, sizeof(uint), MTL::ResourceStorageModeManaged));

        m_command_buffer = m_queue->commandBuffer();

        m_encoder = m_command_buffer->computeCommandEncoder();

        m_encoder->setComputePipelineState(m_pipeline.get());
        m_encoder->setBuffer(bufA.get(), 0, 0);
        m_encoder->setBuffer(bufB.get(), 0, 1);
        m_encoder->setBuffer(bufC.get(), 0, 2);
        m_encoder->setBuffer(bufWidth.get(), 0, 3);

        MTL::Size grid(NROWS, NCOLS, 1);
        MTL::Size threadsPerThreadgroup(NROWS, NCOLS,
                                        1); // Better threadgroup size

        m_encoder->dispatchThreads(grid, threadsPerThreadgroup);
        m_encoder->endEncoding();

        runKernel();

        memcpy(matC, bufC.get()->contents(), sizeof(float) * NROWS * NCOLS);
    }

    bool runKernel() {
        m_command_buffer->commit();
        m_command_buffer->waitUntilCompleted();

        // Check for command buffer errors
        if (m_command_buffer->status() == MTL::CommandBufferStatusError) {
            std::cerr << "Command buffer execution failed" << std::endl;
            if (m_command_buffer->error()) {
                std::cerr << "Error: "
                          << m_command_buffer->error()
                                 ->localizedDescription()
                                 ->utf8String()
                          << std::endl;
            }
            return false;
        }
        return true;
    }
};

int main() {
    AutoreleasePoolGuard guard;

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
        std::cout << "Matrix multiplication "
                     "does not match"
                  << std::endl;
    }

    free(matA);
    free(matB);
    free(matC);
    free(matH);

    return 0;
}
