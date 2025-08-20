#ifndef __METAL_BUFFER__
#define __METAL_BUFFER__

#include <Metal/Metal.hpp>

class MetalContext;

class MetalBuffer {
  private:
    NS::SharedPtr<MTL::Buffer> m_buffer;
    size_t m_buffer_size;

  public:
    MetalBuffer(MetalContext context, size_t buffer_size);
    ~MetalBuffer() = default;
    void fillBuffer(void *src, size_t size);
    NS::SharedPtr<MTL::Buffer> getBuffer();
    void *contents();
};

#endif
