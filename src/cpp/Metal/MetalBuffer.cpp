#include <Metal/MetalBuffer.hpp>
#include <Metal/MetalContext.hpp>

NS::SharedPtr<MTL::Buffer> MetalBuffer::getBuffer() { return m_buffer; }

void MetalBuffer::fillBuffer(void *src, size_t size) {
    if (size > m_buffer_size) {
        throw std::runtime_error(
            "The amount of data being copied is larger than the buffer size!");
    }

    std::memcpy(m_buffer->contents(), src, size);

    m_buffer->didModifyRange(NS::Range::Make(0, size));
}

MetalBuffer::MetalBuffer(MetalContext context, size_t buffer_size) {
    auto device = context.getDevice();
    m_buffer = NS::TransferPtr(
        device->newBuffer(buffer_size, MTL::ResourceStorageModeManaged));
    m_buffer_size = buffer_size;
}

void *MetalBuffer::contents() { return m_buffer->contents(); }
