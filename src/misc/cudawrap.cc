/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "pccl_internal.h"
#include "debug.h"
#include "param.h"
#include "cudawrap.h"
#include <string>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>

// This env var (NCCL_CUMEM_ENABLE) toggles hgMem API usage
NCCL_PARAM(CuMemEnable, "CUMEM_ENABLE", -2);
NCCL_PARAM(LegacyCudaRegister, "LEGACY_CUDA_REGISTER", 0);
// Handle type used for cuMemCreate()
HGmemAllocationHandleType ncclHgMemHandleType = HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

static int ncclHgMemSupported = 0;
static decltype(&hgMemGetHandleForAddressRange) pfn_hgMemGetHandleForAddressRange = nullptr;

bool ncclCudaLaunchBlocking = false;

uint ncclSystemDriverVersionCache = PCCL_UNDEF_DRIVER_VERSION;
uint ncclHggcDriverVersionCache = PCCL_UNDEF_DRIVER_VERSION;

// Determine whether HGMEM & VMM RDMA is supported on this platform
int ncclIsHgMemSupported() {
  HGdevice currentDev;
  int hggcDev;
  int flag = 0;
  ncclResult_t ret = ncclSuccess;
  HGGCCHECKGOTO(hggcGetDevice(&hggcDev), ret, error);
  HGCHECKGOTO(hgDeviceGet(&currentDev, hggcDev), ret, error);
  // Query device to see if HGMEM VMM support is available
  HGCHECKGOTO(hgDeviceGetAttribute(&flag, HG_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, currentDev), ret, error);
  if (!flag) return 0;
error:
  return (ret == ncclSuccess);
}

int ncclCuMemEnable() {
  // NCCL_CUMEM_ENABLE=-2 means auto-detect CUMEM support
  int param = ncclParamCuMemEnable();
  return  param >= 0 ? param : (param == -2 && ncclHgMemSupported);
}

int ncclLegacyCudaRegister() {
  int param = ncclParamLegacyCudaRegister();
  return param;
}

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
static ncclResult_t initResult;

static void initOnceFunc() {
  do {
    const char* val = ncclGetEnv("CUDA_LAUNCH_BLOCKING");
    ncclCudaLaunchBlocking = val!=nullptr && val[0]!=0 && !(val[0]=='0' && val[1]==0);
  } while (0);

  pfn_hgMemGetHandleForAddressRange = (decltype(pfn_hgMemGetHandleForAddressRange))
    dlsym(RTLD_DEFAULT, "hgMemGetHandleForAddressRange");

  // Determine whether we support the cuMem APIs or not
  ncclHgMemSupported = ncclIsHgMemSupported();
  return;
}

ncclResult_t ncclHggcLibraryInit() {
  pthread_once(&initOnceControl, initOnceFunc);
  return initResult;
}

bool ncclHggcDmaBufSupported(int device) {
  if (ncclHggcLibraryInit() != ncclSuccess || !pfn_hgMemGetHandleForAddressRange) return false;
  HGdevice dev;
  int supported = 0;
  return hgDeviceGet(&dev, device) == HGGC_SUCCESS &&
    hgDeviceGetAttribute(&supported, HG_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev) == HGGC_SUCCESS && supported;
}

ncclResult_t ncclHggcGetDmaBufFd(int* fd, void* data, size_t size, unsigned long long flags) {
  *fd = -1;
  if (ncclHggcLibraryInit() != ncclSuccess || !pfn_hgMemGetHandleForAddressRange) return ncclInvalidUsage;
  long hostPageSize = sysconf(_SC_PAGESIZE);
  if (hostPageSize <= 0 || size == 0) return ncclInvalidArgument;
  size_t pageSize = (size_t)hostPageSize;
  uintptr_t addr = (uintptr_t)data;
  if (size > UINTPTR_MAX - addr) return ncclInvalidArgument;
  uintptr_t alignedAddr = addr - addr % pageSize;
  if (size > SIZE_MAX - (addr - alignedAddr) - (pageSize - 1)) return ncclInvalidArgument;
  size_t alignedSize = (size + addr - alignedAddr + pageSize - 1) / pageSize * pageSize;
  HGCHECK(pfn_hgMemGetHandleForAddressRange(fd, (HGdeviceptr)alignedAddr, alignedSize,
                                         HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags));
  return ncclSuccess;
}
