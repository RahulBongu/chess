#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int chal_bridge_printf(const char* format, ...);
int chal_bridge_fflush(FILE* stream);

#ifdef __cplusplus
}
#endif
