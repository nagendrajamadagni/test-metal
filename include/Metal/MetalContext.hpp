#ifndef __METAL_CONTEXT__
#define __METAL_CONTEXT__

#include <Metal/Metal.hpp>

class MetalBuffer;

typedef MTL::Size MetalDim;

class MetalContext {
  private:
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::CommandQueue> m_queue;
    NS::SharedPtr<MTL::Library> m_lib;
    NS::SharedPtr<MTL::Function> m_fn;
    NS::SharedPtr<MTL::ComputePipelineState> m_pipeline;
    MTL::CommandBuffer *m_command_buffer;
    MTL::ComputeCommandEncoder *m_encoder;

  public:
    MetalContext(const char *lib, const char *func);

    ~MetalContext() = default;

    void setBuffer(MetalBuffer buffer, NS::UInteger offset,
                   NS::UInteger position);

    void runKernel(MetalDim gridDim, MetalDim blockDim);

    NS::SharedPtr<MTL::Device> getDevice();
};

#endif
