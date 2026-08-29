#include "device.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace tnccl {
namespace {

__global__ void ringAllReduceKernel(DevWork work) {
  const int rank = work.comm.rank;
  const int nranks = work.comm.nranks;
  const unsigned long long reduceSteps =
      static_cast<unsigned long long>(nranks - 1);

  for (std::uint64_t tile = 0; tile < work.tilesPerSegment; ++tile) {
    const unsigned long long ticket0 =
        tileTicketBase(work.baseTicket, tile, nranks);

    // Reduce-Scatter. On edge r -> r+1, ticket s always describes the same
    // segment at both endpoints. Empty segment tiles still publish the ticket.
    device::sendLocal(
        work, work.send, ringSegmentBehind(rank, 1, nranks), tile, ticket0);

    for (int distance = 2; distance < nranks; ++distance) {
      const unsigned long long recvTicket = ticket0 + distance - 2;
      const unsigned long long sendTicket = ticket0 + distance - 1;
      device::recvReduceSend(
          work, ringSegmentBehind(rank, distance, nranks), tile, recvTicket,
          sendTicket);
    }

    device::recvReduceStore(
        work, rank, tile, ticket0 + reduceSteps - 1);

    // All-Gather. This rank owns segment `rank` after Reduce-Scatter.
    const unsigned long long gatherTicket0 = ticket0 + reduceSteps;
    device::sendLocal(work, work.recv, rank, tile, gatherTicket0);

    for (int distance = 1; distance < nranks - 1; ++distance) {
      const unsigned long long recvTicket = gatherTicket0 + distance - 1;
      const unsigned long long sendTicket = gatherTicket0 + distance;
      device::recvCopySend(
          work, ringSegmentBehind(rank, distance, nranks), tile, recvTicket,
          sendTicket);
    }

    device::recvStore(
        work, ringSegmentBehind(rank, nranks - 1, nranks), tile,
        gatherTicket0 + reduceSteps - 1);
  }
}

bool ticketRangeFits(const DevWork& work) {
  const std::uint64_t perTile = messagesPerTile(work.comm.nranks);
  const std::uint64_t max =
      std::numeric_limits<unsigned long long>::max();
  if (work.tilesPerSegment > max / perTile) {
    return false;
  }
  const std::uint64_t messages = work.tilesPerSegment * perTile;
  // The consumer returns credit at ticket + kSlots, so reserve those trailing
  // values in addition to every ready=ticket+1 publication.
  return messages <= max - (kSlots - 1) &&
         work.baseTicket <= max - messages - (kSlots - 1);
}

}  // namespace

tncclResult launchRingAllReduce(const DevWork& work, cudaStream_t stream) {
  if (work.comm.nranks < 1 || work.comm.rank < 0 ||
      work.comm.rank >= work.comm.nranks) {
    return TNCCL_INVALID_ARGUMENT;
  }
  if (work.tilesPerSegment !=
      tilesPerSegmentForCount(work.count, work.comm.nranks)) {
    return TNCCL_INVALID_ARGUMENT;
  }
  if (work.count != 0 && (work.send == nullptr || work.recv == nullptr)) {
    return TNCCL_INVALID_ARGUMENT;
  }

  if (work.comm.nranks == 1) {
    if (work.count == 0 || work.send == work.recv) {
      return TNCCL_SUCCESS;
    }
    if (work.count >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      return TNCCL_INVALID_ARGUMENT;
    }
    const cudaError_t result = cudaMemcpyAsync(
        work.recv, work.send, static_cast<std::size_t>(work.count) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
    return result == cudaSuccess ? TNCCL_SUCCESS : TNCCL_CUDA_ERROR;
  }

  if (work.comm.inbox == nullptr || work.comm.outbox == nullptr ||
      !ticketRangeFits(work)) {
    return TNCCL_INVALID_ARGUMENT;
  }

  if (cudaGetLastError() != cudaSuccess) return TNCCL_CUDA_ERROR;
  ringAllReduceKernel<<<1, kKernelThreads, 0, stream>>>(work);
  return cudaGetLastError() == cudaSuccess ? TNCCL_SUCCESS
                                            : TNCCL_CUDA_ERROR;
}

}  // namespace tnccl
