#include "viproc/viproc.h"

#define VIPROC_STR_(x) #x
#define VIPROC_STR(x) VIPROC_STR_(x)

extern "C" const char* vp_version(void) {
    return VIPROC_STR(VIPROC_VERSION_MAJOR) "." VIPROC_STR(VIPROC_VERSION_MINOR) "." VIPROC_STR(
        VIPROC_VERSION_PATCH);
}
