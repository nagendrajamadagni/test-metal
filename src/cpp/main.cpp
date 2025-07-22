#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <util.h>

#include <Metal/Metal.hpp>

#define NROWS 1000
#define NCOLS 1000

// Multiply 2 square matrices and get the result
void host_square_matmul(float *mat1, float *mat2, float *result, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0;
            for (int k = 0; k < width; ++k) {
                float num1 = mat1[i * width + k];
                float num2 = mat2[k * width + j];
                sum += num1 * num2;
            }
            int idx = i * width + j;
            result[idx] = sum;
        }
    }
}

int main() {
    static_assert(NROWS == NCOLS,
                  "The matrix dimensions do not form a square matrix!");
    float *A = (float *)malloc(sizeof(float) * NROWS * NCOLS);
    float *B = (float *)malloc(sizeof(float) * NROWS * NCOLS);
    float *C = (float *)malloc(sizeof(float) * NROWS * NCOLS);

    float *d_C = (float *)malloc(sizeof(float) * NROWS * NCOLS);
    populate_matrix(A, NROWS, NCOLS);
    populate_matrix(B, NROWS, NCOLS);

    auto start = std::chrono::high_resolution_clock::now();
    host_square_matmul(A, B, C, NROWS);
    auto end = std::chrono::high_resolution_clock::now();

    auto host_delta =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Create an auto release pool for memory management as garbage collection
    // is disabled

    NS::AutoreleasePool *pool =
        NS::AutoreleasePool::alloc()
            ->init(); // I own the auto-release pool now.

    if (!pool) {
        std::cerr << "Failed to create an auto release pool!" << std::endl;
    }

    // Get the default GPU device
    auto *device =
        MTL::CreateSystemDefaultDevice(); // I own the device object now

    if (!device) {
        std::cerr << "Failed to create a metal device!" << std::endl;
        pool->release();
    }

    // Create buffers on the device, similar to cudaMalloc with cudaMemcpy
    auto *bufA = device->newBuffer(A, sizeof(float) * NROWS * NCOLS,
                                   MTL::ResourceStorageModeManaged);

    if (!bufA) {
        std::cerr << "Failed to create a buffer!" << std::endl;
        device->release();
        pool->release();
    }

    auto *bufB = device->newBuffer(B, sizeof(float) * NROWS * NCOLS,
                                   MTL::ResourceStorageModeManaged);

    if (!bufB) {
        std::cerr << "Failed to create a buffer!" << std::endl;
        bufA->release();
        device->release();
        pool->release();
    }

    auto *bufC = device->newBuffer(sizeof(float) * NROWS * NCOLS,
                                   MTL::ResourceStorageModeManaged);

    if (!bufC) {
        std::cerr << "Failed to create a buffer!" << std::endl;
        bufA->release();
        bufB->release();
        device->release();
        pool->release();
    }

    uint32_t nrows = NROWS;

    auto *bufWidth = device->newBuffer(&nrows, sizeof(uint32_t),
                                       MTL::ResourceStorageModeManaged);

    if (!bufWidth) {
        std::cerr << "Failed to create a buffer!" << std::endl;
        bufA->release();
        bufB->release();
        bufC->release();
        device->release();
        pool->release();
    }

    // I own all these buffers and I need to de-allocate them now

    bufA->didModifyRange(NS::Range::Make(0, sizeof(float) * NROWS * NCOLS));
    bufB->didModifyRange(NS::Range::Make(0, sizeof(float) * NROWS * NCOLS));
    bufWidth->didModifyRange(NS::Range::Make(0, sizeof(uint32_t)));

    NS::Error *error = nullptr;
    MTL::Library *lib = nullptr;

    // Find the library

    auto *path = NS::String::string("build/matmul_kernel.metallib",
                                    NS::UTF8StringEncoding);

    lib = device->newLibrary(path, &error);

    // I own the lib function

    if (!lib) {
        std::cerr << "Failed to load Metal library";

        if (error) {
            std::cerr << ": " << error->localizedDescription()->utf8String();
        }
        std::cerr << std::endl;

        bufA->release();
        bufB->release();
        bufC->release();
        bufWidth->release();
        device->release();
        pool->release();
        free(A);
        free(B);
        free(C);
        free(d_C);
        return -1;
    }

    // Find the function inside the library

    auto *functionName =
        NS::String::string("device_matrix_multiply", NS::UTF8StringEncoding);

    auto *fn = lib->newFunction(functionName);

    // I own the fn pointer

    if (!fn) {
        std::cerr
            << "Failed to find function 'device_matrix_multiply' in library"
            << std::endl;

        lib->release();
        bufA->release();
        bufB->release();
        bufC->release();
        bufWidth->release();
        device->release();
        pool->release();
        free(A);
        free(B);
        free(C);
        free(d_C);
        return -1;
    }

    // Create a compute pipeline

    auto *pipeline = device->newComputePipelineState(fn, &error);

    // I own the pipeline object now

    if (!pipeline) {
        std::cerr << "Failed to create compute pipeline state";

        if (error) {
            std::cerr << ": " << error->localizedDescription()->utf8String();
        }

        std::cerr << std::endl;
        fn->release();
        lib->release();
        bufA->release();
        bufB->release();
        bufC->release();
        bufWidth->release();
        device->release();
        pool->release();
        free(A);
        free(B);
        free(C);
        free(d_C);
        return -1;
    }

    // Encode the commands and dispatch

    auto *queue = device->newCommandQueue();

    if (!queue) {
        std::cerr << "Failed to create a command queue!";

        fn->release();
        lib->release();
        bufA->release();
        bufB->release();
        bufC->release();
        bufWidth->release();
        device->release();
        pool->release();
        free(A);
        free(B);
        free(C);
        free(d_C);
        return -1;
    }

    auto *cmdBuf = queue->commandBuffer();
    auto *enc = cmdBuf->computeCommandEncoder();

    // I only own the queue object here

    enc->setComputePipelineState(pipeline);
    enc->setBuffer(bufA, 0, 0);
    enc->setBuffer(bufB, 0, 1);
    enc->setBuffer(bufC, 0, 2);
    enc->setBuffer(bufWidth, 0, 3);

    // Configure the threads and thread groups manually

    MTL::Size threadsPerThreadGroup = MTL::Size::Make(NROWS, NCOLS, 1);
    MTL::Size numThreadGroups = MTL::Size::Make(1, 1, 1);

    start = std::chrono::high_resolution_clock::now();

    enc->dispatchThreadgroups(numThreadGroups, threadsPerThreadGroup);

    enc->endEncoding();

    cmdBuf->commit();

    cmdBuf->waitUntilCompleted();

    end = std::chrono::high_resolution_clock::now();

    auto device_delta =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (cmdBuf->status() == MTL::CommandBufferStatusError) {
        std::cerr << "Command buffer execution failed" << std::endl;

        if (cmdBuf->error()) {
            std::cerr << "Error: "
                      << cmdBuf->error()->localizedDescription()->utf8String()
                      << std::endl;
        }
    }

    if (cmdBuf->status() == MTL::CommandBufferStatusCompleted) {
        memcpy(d_C, bufC->contents(), sizeof(float) * NROWS * NCOLS);
    }

    if (!compare_matrices(C, d_C, NROWS, NCOLS)) {
        std::cerr << "Device matrix multiplication result does not match host "
                     "matrix multiplication result!"
                  << std::endl;
    } else {
        std::cout << "Device matrix multiplication and host matrix "
                     "multiplication results are equal!"
                  << std::endl;
    }

    std::cout
        << "The time taken to perform matrix multiplication on the host is "
        << host_delta.count()
        << " and the time taken to perform the same operation on the device is "
        << device_delta.count() << std::endl;

    bufA->release();
    bufB->release();
    bufC->release();
    pipeline->release();
    fn->release();
    lib->release();
    queue->release();
    device->release();
    pool->release();
    free(A);
    free(B);
    free(C);
    free(d_C);

    return 0;
}
