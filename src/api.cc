#include "internal.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

std::uint64_t divideRoundUp(std::uint64_t value, std::uint64_t divisor) {
  return value / divisor + (value % divisor != 0);
}

bool rangesOverlap(const float* send, const float* recv, std::size_t count) {
  if (send == recv || count == 0) return false;
  const std::size_t bytes = count * sizeof(float);
  const std::uintptr_t sendBegin = reinterpret_cast<std::uintptr_t>(send);
  const std::uintptr_t recvBegin = reinterpret_cast<std::uintptr_t>(recv);
  if (sendBegin > std::numeric_limits<std::uintptr_t>::max() - bytes ||
      recvBegin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return true;
  }
  return sendBegin < recvBegin + bytes && recvBegin < sendBegin + bytes;
}

}  // namespace

extern "C" tncclResult tncclAllReduceSumF32(
    const float* send, float* recv, std::size_t count, tncclComm_t comm,
    cudaStream_t stream) {
  if (comm == nullptr ||
      (count != 0 && (send == nullptr || recv == nullptr)) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      rangesOverlap(send, recv, count)) {
    return TNCCL_INVALID_ARGUMENT;
  }

  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return TNCCL_CUDA_ERROR;
  if (device != comm->device) return TNCCL_INVALID_ARGUMENT;

  if (comm->streamBound && comm->boundStream != stream) {
    return TNCCL_UNSUPPORTED;
  }

  if (comm->nranks == 1) {
    if (count != 0 && send != recv &&
        cudaMemcpyAsync(recv, send, count * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return TNCCL_CUDA_ERROR;
    }
    if (!comm->streamBound) {
      comm->boundStream = stream;
      comm->streamBound = true;
    }
    return TNCCL_SUCCESS;
  }

  const std::uint64_t maxSegmentElements =
      divideRoundUp(static_cast<std::uint64_t>(count),
                    static_cast<std::uint64_t>(comm->nranks));
  const std::uint64_t tilesPerSegment =
      maxSegmentElements == 0
          ? 1
          : divideRoundUp(maxSegmentElements, tnccl::kChunkElems);
  const std::uint64_t messagesPerTile =
      2ull * static_cast<std::uint64_t>(comm->nranks - 1);
  const std::uint64_t maxTicket =
      std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t trailingCredits = tnccl::kSlots - 1;
  if (comm->nextTicket > maxTicket - trailingCredits ||
      tilesPerSegment >
          (maxTicket - trailingCredits - comm->nextTicket) /
              messagesPerTile) {
    return TNCCL_UNSUPPORTED;
  }

  tnccl::DevWork work{};
  work.comm = comm->dev;
  work.send = send;
  work.recv = recv;
  work.count = static_cast<std::uint64_t>(count);
  work.baseTicket = comm->nextTicket;
  work.tilesPerSegment = tilesPerSegment;

  const tncclResult result = tnccl::launchRingAllReduce(work, stream);
  if (result != TNCCL_SUCCESS) return result;

  comm->nextTicket += tilesPerSegment * messagesPerTile;
  if (!comm->streamBound) {
    comm->boundStream = stream;
    comm->streamBound = true;
  }
  return TNCCL_SUCCESS;
}

extern "C" const char* tncclGetErrorString(tncclResult result) {
  switch (result) {
    case TNCCL_SUCCESS:
      return "success";
    case TNCCL_INVALID_ARGUMENT:
      return "invalid argument";
    case TNCCL_UNSUPPORTED:
      return "unsupported configuration";
    case TNCCL_SYSTEM_ERROR:
      return "system error";
    case TNCCL_CUDA_ERROR:
      return "CUDA error";
    default:
      return "unknown tiny-nccl error";
  }
}
