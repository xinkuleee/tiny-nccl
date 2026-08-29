#include "internal.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace {

tncclResult initializeMailbox(tnccl::Mailbox** mailbox,
                              cudaIpcMemHandle_t* handle) {
  std::unique_ptr<tnccl::Mailbox> initial(
      new (std::nothrow) tnccl::Mailbox{});
  if (!initial) return TNCCL_SYSTEM_ERROR;

  for (std::uint32_t slot = 0; slot < tnccl::kSlots; ++slot) {
    initial->slots[slot].ready = 0;
    initial->slots[slot].credit = slot;
  }

  cudaError_t error =
      cudaMalloc(reinterpret_cast<void**>(mailbox), sizeof(tnccl::Mailbox));
  if (error != cudaSuccess) return TNCCL_CUDA_ERROR;

  error = cudaMemcpy(*mailbox, initial.get(), sizeof(tnccl::Mailbox),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) error = cudaIpcGetMemHandle(handle, *mailbox);
  if (error != cudaSuccess) {
    cudaFree(*mailbox);
    *mailbox = nullptr;
    std::memset(handle, 0, sizeof(*handle));
    return TNCCL_CUDA_ERROR;
  }
  return TNCCL_SUCCESS;
}

tncclResult firstPeerFailure(const std::vector<tnccl::PeerMeta>& peers) {
  for (const tnccl::PeerMeta& peer : peers) {
    if (peer.status < TNCCL_SUCCESS || peer.status > TNCCL_CUDA_ERROR) {
      return TNCCL_SYSTEM_ERROR;
    }
    if (peer.status != TNCCL_SUCCESS) {
      return static_cast<tncclResult>(peer.status);
    }
  }
  return TNCCL_SUCCESS;
}

bool devicesAreUnique(const std::vector<tnccl::PeerMeta>& peers) {
  for (std::size_t i = 0; i < peers.size(); ++i) {
    if (peers[i].device < 0) return false;
    for (std::size_t j = 0; j < i; ++j) {
      if (peers[i].device == peers[j].device) return false;
    }
  }
  return true;
}

tncclResult openNextMailbox(tncclComm* comm,
                            const tnccl::PeerMeta& nextPeer) {
  if (nextPeer.device < 0 || nextPeer.device == comm->device) {
    return TNCCL_UNSUPPORTED;
  }

  int canAccess = 0;
  cudaError_t error =
      cudaDeviceCanAccessPeer(&canAccess, comm->device, nextPeer.device);
  if (error != cudaSuccess) return TNCCL_CUDA_ERROR;
  if (canAccess == 0) return TNCCL_UNSUPPORTED;

  int nativeAtomics = 0;
  error = cudaDeviceGetP2PAttribute(&nativeAtomics,
                                    cudaDevP2PAttrNativeAtomicSupported,
                                    comm->device, nextPeer.device);
  if (error != cudaSuccess) return TNCCL_CUDA_ERROR;
  if (nativeAtomics == 0) return TNCCL_UNSUPPORTED;

  error = cudaIpcOpenMemHandle(
      reinterpret_cast<void**>(&comm->remoteNextInbox),
      nextPeer.mailboxHandle, cudaIpcMemLazyEnablePeerAccess);
  return error == cudaSuccess ? TNCCL_SUCCESS : TNCCL_CUDA_ERROR;
}

void closeRemoteMailbox(tncclComm* comm, tncclResult* result) {
  if (comm->remoteNextInbox == nullptr) return;
  if (cudaIpcCloseMemHandle(comm->remoteNextInbox) != cudaSuccess &&
      *result == TNCCL_SUCCESS) {
    *result = TNCCL_CUDA_ERROR;
  }
  comm->remoteNextInbox = nullptr;
  comm->dev.outbox = nullptr;
}

void freeLocalMailbox(tncclComm* comm, tncclResult* result) {
  if (comm->localInbox == nullptr) return;
  if (cudaFree(comm->localInbox) != cudaSuccess &&
      *result == TNCCL_SUCCESS) {
    *result = TNCCL_CUDA_ERROR;
  }
  comm->localInbox = nullptr;
  comm->dev.inbox = nullptr;
}

// Once an IPC handle has been published, a control-plane failure means we can
// no longer prove that every peer closed its import. Close our own import but
// deliberately keep the exported allocation alive until process exit.
void abandonPublishedInit(tncclComm* comm) {
  tncclResult ignored = TNCCL_SUCCESS;
  closeRemoteMailbox(comm, &ignored);
  tnccl::bootstrapClose(&comm->bootstrap);
}

}  // namespace

extern "C" tncclResult tncclCommInitRank(tncclComm_t* output, int nranks,
                                            tncclUniqueId id, int rank) {
  if (output == nullptr || nranks <= 0 || rank < 0 || rank >= nranks) {
    return TNCCL_INVALID_ARGUMENT;
  }
  *output = nullptr;

  std::unique_ptr<tncclComm> comm(new (std::nothrow) tncclComm);
  if (!comm) return TNCCL_SYSTEM_ERROR;

  comm->rank = rank;
  comm->nranks = nranks;
  if (cudaGetDevice(&comm->device) != cudaSuccess) return TNCCL_CUDA_ERROR;
  comm->dev.rank = rank;
  comm->dev.nranks = nranks;

  if (nranks == 1) {
    *output = comm.release();
    return TNCCL_SUCCESS;
  }

  tncclResult result =
      tnccl::bootstrapOpen(&comm->bootstrap, id, nranks, rank);
  if (result != TNCCL_SUCCESS) return result;

  tnccl::PeerMeta local{};
  local.device = comm->device;
  local.status = initializeMailbox(&comm->localInbox, &local.mailboxHandle);
  comm->dev.inbox = comm->localInbox;

  std::vector<tnccl::PeerMeta> peers;
  try {
    peers.resize(static_cast<std::size_t>(nranks));
  } catch (const std::bad_alloc&) {
    abandonPublishedInit(comm.get());
    return TNCCL_SYSTEM_ERROR;
  }

  result = tnccl::bootstrapAllGather(&comm->bootstrap, &local, peers.data(),
                                     sizeof(local));
  if (result != TNCCL_SUCCESS) {
    abandonPublishedInit(comm.get());
    return result;
  }

  result = firstPeerFailure(peers);
  if (result == TNCCL_SUCCESS && !devicesAreUnique(peers)) {
    result = TNCCL_UNSUPPORTED;
  }
  if (result == TNCCL_SUCCESS) {
    const int next = (rank + 1) % nranks;
    result = openNextMailbox(comm.get(), peers[static_cast<std::size_t>(next)]);
    if (result == TNCCL_SUCCESS) comm->dev.outbox = comm->remoteNextInbox;
  }

  tnccl::PeerMeta connectionStatus{};
  connectionStatus.status = result;
  connectionStatus.device = comm->device;
  std::vector<tnccl::PeerMeta> connectionStatuses;
  try {
    connectionStatuses.resize(static_cast<std::size_t>(nranks));
  } catch (const std::bad_alloc&) {
    abandonPublishedInit(comm.get());
    return TNCCL_SYSTEM_ERROR;
  }

  const tncclResult gatherResult = tnccl::bootstrapAllGather(
      &comm->bootstrap, &connectionStatus, connectionStatuses.data(),
      sizeof(connectionStatus));
  if (gatherResult != TNCCL_SUCCESS) {
    abandonPublishedInit(comm.get());
    return gatherResult;
  }
  result = firstPeerFailure(connectionStatuses);

  // All ranks reach the same barrier after the status gather, regardless of
  // success. This both completes the ready phase and orders failed cleanup.
  const tncclResult barrierResult =
      tnccl::bootstrapBarrier(&comm->bootstrap);
  if (barrierResult != TNCCL_SUCCESS) {
    abandonPublishedInit(comm.get());
    return barrierResult;
  }

  if (result != TNCCL_SUCCESS) {
    tncclResult ignored = TNCCL_SUCCESS;
    closeRemoteMailbox(comm.get(), &ignored);
    const tncclResult cleanupBarrier =
        tnccl::bootstrapBarrier(&comm->bootstrap);
    if (cleanupBarrier == TNCCL_SUCCESS) {
      freeLocalMailbox(comm.get(), &ignored);
    }
    tnccl::bootstrapClose(&comm->bootstrap);
    return cleanupBarrier == TNCCL_SUCCESS ? result : cleanupBarrier;
  }

  *output = comm.release();
  return TNCCL_SUCCESS;
}

extern "C" tncclResult tncclCommDestroy(tncclComm_t comm) {
  if (comm == nullptr) return TNCCL_INVALID_ARGUMENT;

  int originalDevice = -1;
  if (cudaGetDevice(&originalDevice) != cudaSuccess) {
    // Consume the host handle consistently. CUDA resources are intentionally
    // left to context/process teardown because their owning device is unknown.
    tnccl::bootstrapClose(&comm->bootstrap);
    delete comm;
    return TNCCL_CUDA_ERROR;
  }
  if (originalDevice != comm->device &&
      cudaSetDevice(comm->device) != cudaSuccess) {
    tnccl::bootstrapClose(&comm->bootstrap);
    delete comm;
    return TNCCL_CUDA_ERROR;
  }

  tncclResult result = TNCCL_SUCCESS;
  if (comm->streamBound &&
      cudaStreamSynchronize(comm->boundStream) != cudaSuccess) {
    // The kernel may still own either IPC mapping. Break the control
    // connection so peers take their bounded error paths, but retain CUDA
    // resources until context/process teardown.
    tnccl::bootstrapClose(&comm->bootstrap);
    delete comm;
    return TNCCL_CUDA_ERROR;
  }

  bool firstBarrierComplete = comm->nranks == 1;
  if (comm->nranks > 1) {
    // Destroy is collective. First establish that every bound stream has
    // finished before any rank invalidates a mapping still used by a peer.
    const tncclResult barrierResult =
        tnccl::bootstrapBarrier(&comm->bootstrap);
    if (barrierResult != TNCCL_SUCCESS && result == TNCCL_SUCCESS) {
      result = barrierResult;
    }
    firstBarrierComplete = barrierResult == TNCCL_SUCCESS;
  }

  // Continue local teardown after a failed barrier. Peers observe connection
  // close or their own timeout; the exported allocation is retained below
  // unless both collective teardown barriers complete.
  closeRemoteMailbox(comm, &result);

  bool importersClosed = comm->nranks == 1;
  if (comm->nranks > 1) {
    // Only after every importer has closed its mapping may exporters free the
    // allocations backing those mappings.
    const tncclResult barrierResult =
        tnccl::bootstrapBarrier(&comm->bootstrap);
    if (barrierResult != TNCCL_SUCCESS && result == TNCCL_SUCCESS) {
      result = barrierResult;
    }
    importersClosed = firstBarrierComplete &&
                      barrierResult == TNCCL_SUCCESS;
  }

  if (importersClosed) {
    freeLocalMailbox(comm, &result);
  }
  tnccl::bootstrapClose(&comm->bootstrap);

  if (originalDevice != comm->device &&
      cudaSetDevice(originalDevice) != cudaSuccess &&
      result == TNCCL_SUCCESS) {
    result = TNCCL_CUDA_ERROR;
  }

  delete comm;
  return result;
}
