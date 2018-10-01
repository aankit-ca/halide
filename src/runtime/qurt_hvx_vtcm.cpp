#include "runtime_internal.h"
#include "HalideRuntimeQurt.h"
#include "mini_qurt.h"
#include "mini_qurt_vtcm.h"

using namespace Halide::Runtime::Internal::Qurt;

extern "C" {

WEAK void* halide_vtcm_malloc(void *user_context, int size) {
    return HAP_request_VTCM(size, 1);
}

WEAK void halide_vtcm_free(void *user_context, void *addr) {
    HAP_release_VTCM(addr);
}

// __attribute__((always_inline))
// WEAK int halide_scatter_release(void *ptr) {
//     __asm__ volatile("vmem(%0 + #0):scatter_release\n; v1 = vmem(%0 + #0)\n"
//                      : "+m" (*(char *)ptr) : : "v1");
//     return 0;
// }

}
