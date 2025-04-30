#ifndef ED25519_REF10_H
#define ED25519_REF10_H

#include <stdint.h>

// Reference Ed25519 scalar-base multiplication prototype
// r: output 32-byte public key
// a: input 32-byte secret scalar
void ge_scalarmult_base(unsigned char *r, const unsigned char *a);

#endif // ED25519_REF10_H
