#include <Metal/MetalBuffer.hpp>
#include <Metal/MetalContext.hpp>
#include <iostream>

MetalContext::MetalContext(const char *lib, const char *func) {
    m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    m_queue = NS::TransferPtr(m_device->newCommandQueue());

    NS::Error *error = nullptr;
    auto *library = NS::String::string(lib, NS::UTF8StringEncoding);
    m_lib = NS::TransferPtr(m_device->newLibrary(library, &error));

    MTL::ComputePipelineReflection *reflection = nullptr;

    if (!m_lib) {
        std::cerr << "Failed to create library "
                  << error->localizedDescription()->utf8String() << std::endl;
        throw std::runtime_error("Failed to create Metal library");
    }

    auto *function = NS::String::string(func, NS::UTF8StringEncoding);
    m_fn = NS::TransferPtr(m_lib->newFunction(function));

    if (!m_fn) {
        throw std::runtime_error("Failed to find function in library");
    }

    m_pipeline = NS::TransferPtr(m_device->newComputePipelineState(
        m_fn.get(),
        MTL::PipelineOptionArgumentInfo | MTL::PipelineOptionBufferTypeInfo,
        &reflection, &error));

    m_command_buffer = m_queue->commandBuffer();

    m_encoder = m_command_buffer->computeCommandEncoder();

    m_encoder->setComputePipelineState(m_pipeline.get());
}

void MetalContext::setBuffer(MetalBuffer buffer, NS::UInteger offset,
                             NS::UInteger position) {
    m_encoder->setBuffer(buffer.getBuffer().get(), offset, position);
}

NS::SharedPtr<MTL::Device> MetalContext::getDevice() { return m_device; }

void MetalContext::runKernel(MetalDim gridDim, MetalDim blockDim) {
    m_encoder->dispatchThreads(gridDim, blockDim);
    m_encoder->endEncoding();

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
        throw std::runtime_error("Failed to run the metal kernel!");
    }
}
