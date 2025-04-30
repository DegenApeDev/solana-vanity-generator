#include "ed25519_ref10.h"
#include <string.h>

// Stub implementation: replace with full ref10 scalar-base multiplication.
void ge_scalarmult_base(unsigned char *r, const unsigned char *a) {
    // TODO: implement using SUPERCOP ref10 code
    memset(r, 0, 32);
}
