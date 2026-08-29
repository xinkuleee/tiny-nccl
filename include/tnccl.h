#pragma once

#include <stddef.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tncclComm* tncclComm_t;

typedef struct {
  unsigned char bytes[128];
} tncclUniqueId;

typedef enum {
  TNCCL_SUCCESS = 0,
  TNCCL_INVALID_ARGUMENT = 1,
  TNCCL_UNSUPPORTED = 2,
  TNCCL_SYSTEM_ERROR = 3,
  TNCCL_CUDA_ERROR = 4
} tncclResult;

tncclResult tncclGetUniqueId(tncclUniqueId* id);

// Blocking collective initialization for nranks > 1. Every rank must call this
// once with the same id/nranks, a unique rank, and its target CUDA device set as
// current. The communicator captures that device and is single-host-thread.
tncclResult tncclCommInitRank(
    tncclComm_t* comm, int nranks, tncclUniqueId id, int rank);

// Enqueues FP32 SUM AllReduce on stream; success does not mean completion.
// All ranks must use the same collective order and count. The first call binds
// comm to stream; send/recv must remain valid through stream completion, and
// stream must remain valid until tncclCommDestroy returns. Only exact in-place
// or fully non-overlapping buffers are supported. Runtime contract violations
// may hang: v0.1 has no GPU-visible abort mechanism.
tncclResult tncclAllReduceSumF32(
    const float* send, float* recv, size_t count, tncclComm_t comm,
    cudaStream_t stream);

// Blocking collective for nranks > 1. Every rank must participate. It
// synchronizes the bound stream and performs ordered CUDA IPC teardown. On
// It always consumes comm, including on error; do not reuse comm or destroy it
// twice. If safe cross-rank teardown cannot be proved, CUDA allocations may be
// intentionally retained until CUDA context/process exit.
// The bound stream must remain alive until this function returns.
tncclResult tncclCommDestroy(tncclComm_t comm);

const char* tncclGetErrorString(tncclResult result);

#ifdef __cplusplus
}
#endif
