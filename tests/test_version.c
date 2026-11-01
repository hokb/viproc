#include <stdio.h>
#include <string.h>

#include "numpy2x/numpy2x.h"

int main(void) {
    const char* v = nx_version();
    if (v == NULL || strcmp(v, "0.1.0") != 0) {
        fprintf(stderr, "unexpected version: %s\n", v ? v : "(null)");
        return 1;
    }
    return 0;
}
