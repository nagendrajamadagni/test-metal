#include <metal_stdlib>
using namespace metal;

// vector_add kernel: C = A + B
kernel void device_matrix_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 id [[thread_position_in_grid]])
{
    uint row = id.y;
    uint col = id.x;

    float sum = 0;

    for (uint k = 0; k < width; ++k) {
        sum += A[row * width + k] * B[k * width + col];
    }
    C[row * width + col] = sum;
}
