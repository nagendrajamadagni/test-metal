#include <Metal/Metal.hpp>

#ifndef __AUTORELEASE_POOL_GUARD__
#define __AUTORELEASE_POOL_GUARD__

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

#endif
