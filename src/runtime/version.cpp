#include "numpy2x/numpy2x.h"

#define NUMPY2X_STR_(x) #x
#define NUMPY2X_STR(x) NUMPY2X_STR_(x)

extern "C" const char* nx_version(void) {
    return NUMPY2X_STR(NUMPY2X_VERSION_MAJOR) "." NUMPY2X_STR(
        NUMPY2X_VERSION_MINOR) "." NUMPY2X_STR(NUMPY2X_VERSION_PATCH);
}
