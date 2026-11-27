/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_NET_PROFILER_H_
#define PCCL_NET_PROFILER_H_

#ifdef NCCL_ENABLE_NET_PROFILING
#include "pccl_internal.h"
#include "plugin/pccl_profiler.h"
#include "plugin/profiler/net_ib_v1.h"
#include "plugin/profiler/net_socket_v1.h"

ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData);

static inline ncclResult_t pcclProfilerNetStart(void** handle, void* parent, int64_t id, void* data) {
  if (parent && !*handle) NCCLCHECK(ncclProfilerCallback(handle, ncclProfilerNetEventStart, parent, id, data));
  return ncclSuccess;
}

static inline ncclResult_t pcclProfilerNetStop(void** handle) {
  if (*handle) {
    NCCLCHECK(ncclProfilerCallback(handle, ncclProfilerNetEventStop, nullptr, 0, nullptr));
    *handle = nullptr;
  }
  return ncclSuccess;
}
#endif

#endif
