#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

namespace profiling {
// Host launch ranges identify asynchronous GPU work without synchronizing it.
class Range {
public:
    explicit Range(const char* name) {
#if defined(__HIP_PLATFORM_AMD__)
        roctxRangePushA(name);
#endif
    }
    ~Range() {
#if defined(__HIP_PLATFORM_AMD__)
        roctxRangePop();
#endif
    }
    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
};
} // namespace profiling
