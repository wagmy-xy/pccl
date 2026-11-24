/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "ibvwrap.h"
#include <errno.h>

// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
ncclResult_t ncclIbGdrSupport() {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    // Check for the alixpu_peer_mem module being loaded
    moduleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/alixpu_peer_mem/version") ||
                   KNL_MODULE_LOADED("/sys/module/alixpu_peermem/version");
  }
  if (moduleLoaded == 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t ncclIbPeerMemSupport() {
  return ncclIbGdrSupport();
}

ncclResult_t ncclIbDmaBufSupport(int dev) {
  if (dev < 0 || dev >= MAX_IB_DEVS) return ncclSystemError;
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  if (ibDev->dmaBufSupported == 1) return ncclSuccess;
  if (ibDev->dmaBufSupported == 0) return ncclSystemError;

  if (ibDev->context == NULL) return ncclSystemError;

  struct ibv_pd* pd = NULL;
  ncclResult_t res = wrap_ibv_alloc_pd(&pd, ibDev->context);
  if (res != ncclSuccess || pd == NULL) {
    ibDev->dmaBufSupported = 0;
    return ncclSystemError;
  }

  errno = 0;
  struct ibv_mr* mr = wrap_direct_ibv_reg_dmabuf_mr(pd, 0 /*offset*/, 0 /*len*/, 0 /*iova*/, -1 /*fd*/, 0 /*access*/);
  bool devFail = (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  // An invalid fd should fail, but release an MR if a provider unexpectedly returns one.
  if (mr != NULL) res = wrap_ibv_dereg_mr(mr);
  ncclResult_t deallocResult = wrap_ibv_dealloc_pd(pd);

  if (devFail || res != ncclSuccess || deallocResult != ncclSuccess) {
    ibDev->dmaBufSupported = 0;
    return ncclSystemError;
  }
  ibDev->dmaBufSupported = 1;
  return ncclSuccess;
}
