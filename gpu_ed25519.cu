// gpu_ed25519.cu
/*
 * Ed25519 public-key derivation kernel using ref10 code.
 * Requires ed25519_ref10.h/.c from the SUPERCOP reference implementation.
 */
extern "C" {
    #include "ed25519_ref10.h"    // provide ge_scalarmult_base
}

#include <stdint.h>

extern "C" __global__ void derive_pubkeys(const uint8_t *seeds,
                                           uint8_t *pubs,
                                           int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const uint8_t *seed = &seeds[idx * 32];
        uint8_t *pub = &pubs[idx * 32];
        // perform scalar-base multiplication
        ge_scalarmult_base(pub, seed);
    }
}
