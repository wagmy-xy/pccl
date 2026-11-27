/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include <sys/socket.h>
#include "pccl_internal.h"
#include "pccl_net_internal.h"
#include "checks.h"

#define EIC_PLUGIN_NAME "accl-lite"
extern int nccl_use_u2mm_host;
extern int nccl_use_u2mm_dev;
extern int nccl_eic_gpu_offset;
typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

ncclResult_t ncclNetInit(ncclComm_t comm);
ncclResult_t ncclNetFinalize(ncclComm_t comm);
int ncclNetVersion(ncclComm_t comm);
bool ncclEicNet(ncclNet_t* net);

// Test whether the current GPU support GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#ifdef NCCL_ENABLE_NET_PROFILING
// Internal overloads carry Proxy step handles without changing the NET v4/v5/v6 ABI.
ncclResult_t ncclIbIsend(void*, void*, int, int, void*, void*, void**);
ncclResult_t ncclIbIrecv(void*, int, void**, int*, int*, void**, void**, void**);
ncclResult_t ncclNetSocketIsend(void*, void*, int, int, void*, void*, void**);
ncclResult_t ncclNetSocketIrecv(void*, int, void**, int*, int*, void**, void**, void**);
#endif

static inline ncclResult_t pcclNetIsend(ncclNet_t* net, void* comm, void* data, int size, int tag,
                                      void* mhandle, void* phandle, void** request) {
#ifdef NCCL_ENABLE_NET_PROFILING
  if (net == &ncclNetIb) return ncclIbIsend(comm, data, size, tag, mhandle, phandle, request);
  if (net == &ncclNetSocket) return ncclNetSocketIsend(comm, data, size, tag, mhandle, phandle, request);
#endif
  return net->isend(comm, data, size, tag, mhandle, request);
}

static inline ncclResult_t pcclNetIrecv(ncclNet_t* net, void* comm, int n, void** data, int* sizes, int* tags,
                                      void** mhandles, void** phandles, void** request) {
#ifdef NCCL_ENABLE_NET_PROFILING
  if (net == &ncclNetIb) return ncclIbIrecv(comm, n, data, sizes, tags, mhandles, phandles, request);
  if (net == &ncclNetSocket) return ncclNetSocketIrecv(comm, n, data, sizes, tags, mhandles, phandles, request);
#endif
  return net->irecv(comm, n, data, sizes, tags, mhandles, request);
}

#endif
