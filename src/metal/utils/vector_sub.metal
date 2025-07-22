#include <metal_stdlib>
using namespace metal;

// vector_add kernel: C = A + B
kernel void vector_sub(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] + B[id];
}
