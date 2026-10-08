/* Public C ABI of the viproc runtime. Everything exported to language
 * bridges (Python first) goes through this header. */
#ifndef VIPROC_VIPROC_H
#define VIPROC_VIPROC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the runtime version as "MAJOR.MINOR.PATCH". */
const char* vp_version(void);

#ifdef __cplusplus
}
#endif

#endif /* VIPROC_VIPROC_H */
