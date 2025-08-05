// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTIRATTRS_H
#define TTMLIR_C_TTIRATTRS_H

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTIRThreadAttrGet(
    MlirContext ctx, uint32_t threadType, MlirAttribute kernelSymbol);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTIRATTRS_H
