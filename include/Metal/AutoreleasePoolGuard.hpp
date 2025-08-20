#include <Metal/Metal.hpp>
#include <iostream>

#ifndef __AUTORELEASE_POOL_GUARD__
#define __AUTORELEASE_POOL_GUARD__

class AutoreleasePoolGuard {
  private:
    NS::AutoreleasePool *pool;

  public:
    AutoreleasePoolGuard() : pool(NS::AutoreleasePool::alloc()->init()) {
        std::cout << "Calling constructor for arp" << std::endl;
    }
    ~AutoreleasePoolGuard() {
        std::cout << "Calling destructor for arp" << std::endl;
        if (pool) {
            pool->release();
        }
    }
    // Prevent copying of autorelease pools
    AutoreleasePoolGuard(const AutoreleasePoolGuard &) = delete;
    AutoreleasePoolGuard &operator=(const AutoreleasePoolGuard &) = delete;
};

#endif
