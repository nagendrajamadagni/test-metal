#include <Metal/AutoreleasePoolGuard.hpp>
#include <Metal/MetalBuffer.hpp>
#include <Metal/MetalContext.hpp>
#include <iostream>

NS::SharedPtr<MTL::Buffer> MetalBuffer::getBuffer() {

    AutoreleasePoolGuard guard;
    return m_buffer;
}

void MetalBuffer::fillBuffer(void *src, size_t size) {
    AutoreleasePoolGuard guard;
    if (size > m_buffer_size) {
        throw std::runtime_error(
            "The amount of data being copied is larger than the buffer size!");
    }

    std::memcpy(m_buffer->contents(), src, size);

    m_buffer->didModifyRange(NS::Range::Make(0, size));
}

MetalBuffer::MetalBuffer(MetalContext context, size_t buffer_size) {
    AutoreleasePoolGuard guard;
    std::cout << "Inside the Metal Buffer Constructor" << std::endl;
    auto device = context.getDevice();
    m_buffer = NS::TransferPtr(
        device->newBuffer(buffer_size, MTL::ResourceStorageModeManaged));
    m_buffer_size = buffer_size;
}

MetalBuffer::~MetalBuffer() {
    AutoreleasePoolGuard guard;
    std::cout << "Inside the Metal buffer destructor" << std::endl;
}

void *MetalBuffer::contents() {

    AutoreleasePoolGuard guard;
    return m_buffer->contents();
}
