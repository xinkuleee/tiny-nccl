#pragma once

#include "internal.h"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>

namespace tnccl {

static_assert(kSlots >= 2,
              "a fused receive/forward needs distinct FIFO slots");
static_assert(kChunkElems > 0, "a FIFO payload must hold at least one float");

struct ElementSpan {
  std::uint64_t offset;
  std::uint64_t count;
};

#define TNCCL_HOST_DEVICE_INLINE __host__ __device__ __forceinline__

TNCCL_HOST_DEVICE_INLINE constexpr std::uint64_t divUp(
    std::uint64_t value, std::uint64_t divisor) {
  return value / divisor + (value % divisor != 0);
}

TNCCL_HOST_DEVICE_INLINE constexpr ElementSpan segmentSpan(
    std::uint64_t count, int nranks, int segment) {
  const std::uint64_t ranks = static_cast<std::uint64_t>(nranks);
  const std::uint64_t shortCount = count / ranks;
  const std::uint64_t longSegments = count % ranks;
  const std::uint64_t index = static_cast<std::uint64_t>(segment);
  const std::uint64_t prefixLong = index < longSegments ? index : longSegments;
  return {index * shortCount + prefixLong,
          shortCount + (index < longSegments ? 1u : 0u)};
}

TNCCL_HOST_DEVICE_INLINE constexpr std::uint64_t tilesPerSegmentForCount(
    std::uint64_t count, int nranks) {
  const std::uint64_t maxSegment =
      divUp(count, static_cast<std::uint64_t>(nranks));
  const std::uint64_t tiles = divUp(maxSegment, kChunkElems);
  return tiles == 0 ? 1 : tiles;
}

TNCCL_HOST_DEVICE_INLINE constexpr ElementSpan segmentTileSpan(
    std::uint64_t count, int nranks, int segment, std::uint64_t tile) {
  const ElementSpan segmentBounds = segmentSpan(count, nranks, segment);
  const std::uint64_t segmentTiles = divUp(segmentBounds.count, kChunkElems);
  if (tile >= segmentTiles) {
    return {segmentBounds.offset + segmentBounds.count, 0};
  }

  const std::uint64_t tileOffset = tile * kChunkElems;
  const std::uint64_t remaining = segmentBounds.count - tileOffset;
  return {segmentBounds.offset + tileOffset,
          remaining < kChunkElems ? remaining : kChunkElems};
}

TNCCL_HOST_DEVICE_INLINE constexpr int ringSegmentBehind(
    int rank, int distance, int nranks) {
  int segment = rank - distance;
  segment %= nranks;
  return segment < 0 ? segment + nranks : segment;
}

TNCCL_HOST_DEVICE_INLINE constexpr std::uint64_t messagesPerTile(int nranks) {
  return 2u * static_cast<std::uint64_t>(nranks - 1);
}

TNCCL_HOST_DEVICE_INLINE constexpr std::uint64_t tileTicketBase(
    std::uint64_t baseTicket, std::uint64_t tile, int nranks) {
  return baseTicket + tile * messagesPerTile(nranks);
}

namespace device {

__device__ __forceinline__ unsigned long long loadAcquireSystem(
    unsigned long long* word) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> atomic(*word);
  return atomic.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void storeReleaseSystem(
    unsigned long long* word, unsigned long long value) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> atomic(*word);
  atomic.store(value, cuda::memory_order_release);
}

__device__ __forceinline__ void spinPause() {
#if __CUDA_ARCH__ >= 700
  __nanosleep(64);
#endif
}

__device__ __forceinline__ Slot* acquireSendSlot(
    Mailbox* outbox, unsigned long long ticket) {
  Slot* slot = &outbox->slots[ticket % kSlots];
  if (threadIdx.x == 0) {
    while (loadAcquireSystem(&slot->credit) != ticket) {
      spinPause();
    }
  }
  __syncthreads();
  return slot;
}

__device__ __forceinline__ Slot* acquireRecvSlot(
    Mailbox* inbox, unsigned long long ticket) {
  Slot* slot = &inbox->slots[ticket % kSlots];
  if (threadIdx.x == 0) {
    while (loadAcquireSystem(&slot->ready) != ticket + 1) {
      spinPause();
    }
  }
  // The block barrier carries thread 0's acquire to the worker threads. Payload
  // loads below are volatile as well, so a reused inbox line cannot be served
  // from a worker's stale L1 state.
  __syncthreads();
  return slot;
}

__device__ __forceinline__ void publishSend(
    Slot* slot, unsigned long long ticket) {
  // Every worker may have written peer memory, so every worker performs the
  // system fence. The barrier then puts all of those fences before thread 0's
  // release publication.
  cuda::atomic_thread_fence(
      cuda::memory_order_release, cuda::thread_scope_system);
  __syncthreads();
  if (threadIdx.x == 0) {
    storeReleaseSystem(&slot->ready, ticket + 1);
  }
  __syncthreads();
}

__device__ __forceinline__ void releaseRecv(
    Slot* slot, unsigned long long ticket) {
  // No producer may overwrite this slot until every worker has finished its
  // payload load.
  __syncthreads();
  if (threadIdx.x == 0) {
    storeReleaseSystem(&slot->credit, ticket + kSlots);
  }
  __syncthreads();
}

__device__ __forceinline__ void releaseRecvAndPublishSend(
    Slot* recvSlot, unsigned long long recvTicket, Slot* sendSlot,
    unsigned long long sendTicket) {
  // The workers have both consumed the receive slot and written the remote send
  // slot. Fence each worker's peer stores before publishing either transition.
  cuda::atomic_thread_fence(
      cuda::memory_order_release, cuda::thread_scope_system);
  __syncthreads();
  if (threadIdx.x == 0) {
    storeReleaseSystem(&recvSlot->credit, recvTicket + kSlots);
    storeReleaseSystem(&sendSlot->ready, sendTicket + 1);
  }
  __syncthreads();
}

__device__ __forceinline__ void sendLocal(
    const DevWork& work, const float* source, int segment, std::uint64_t tile,
    unsigned long long ticket) {
  const ElementSpan span =
      segmentTileSpan(work.count, work.comm.nranks, segment, tile);
  Slot* sendSlot = acquireSendSlot(work.comm.outbox, ticket);

  for (std::uint64_t i = threadIdx.x; i < span.count; i += blockDim.x) {
    sendSlot->payload[i] = source[span.offset + i];
  }
  publishSend(sendSlot, ticket);
}

__device__ __forceinline__ void recvReduceSend(
    const DevWork& work, int segment, std::uint64_t tile,
    unsigned long long recvTicket, unsigned long long sendTicket) {
  const ElementSpan span =
      segmentTileSpan(work.count, work.comm.nranks, segment, tile);
  Slot* recvSlot = acquireRecvSlot(work.comm.inbox, recvTicket);
  Slot* sendSlot = acquireSendSlot(work.comm.outbox, sendTicket);
  const volatile float* received = recvSlot->payload;

  for (std::uint64_t i = threadIdx.x; i < span.count; i += blockDim.x) {
    sendSlot->payload[i] = received[i] + work.send[span.offset + i];
  }
  releaseRecvAndPublishSend(
      recvSlot, recvTicket, sendSlot, sendTicket);
}

__device__ __forceinline__ void recvReduceStore(
    const DevWork& work, int segment, std::uint64_t tile,
    unsigned long long recvTicket) {
  const ElementSpan span =
      segmentTileSpan(work.count, work.comm.nranks, segment, tile);
  Slot* recvSlot = acquireRecvSlot(work.comm.inbox, recvTicket);
  const volatile float* received = recvSlot->payload;

  for (std::uint64_t i = threadIdx.x; i < span.count; i += blockDim.x) {
    work.recv[span.offset + i] =
        received[i] + work.send[span.offset + i];
  }
  releaseRecv(recvSlot, recvTicket);
}

__device__ __forceinline__ void recvCopySend(
    const DevWork& work, int segment, std::uint64_t tile,
    unsigned long long recvTicket, unsigned long long sendTicket) {
  const ElementSpan span =
      segmentTileSpan(work.count, work.comm.nranks, segment, tile);
  Slot* recvSlot = acquireRecvSlot(work.comm.inbox, recvTicket);
  Slot* sendSlot = acquireSendSlot(work.comm.outbox, sendTicket);
  const volatile float* received = recvSlot->payload;

  for (std::uint64_t i = threadIdx.x; i < span.count; i += blockDim.x) {
    const float value = received[i];
    work.recv[span.offset + i] = value;
    sendSlot->payload[i] = value;
  }
  releaseRecvAndPublishSend(
      recvSlot, recvTicket, sendSlot, sendTicket);
}

__device__ __forceinline__ void recvStore(
    const DevWork& work, int segment, std::uint64_t tile,
    unsigned long long recvTicket) {
  const ElementSpan span =
      segmentTileSpan(work.count, work.comm.nranks, segment, tile);
  Slot* recvSlot = acquireRecvSlot(work.comm.inbox, recvTicket);
  const volatile float* received = recvSlot->payload;

  for (std::uint64_t i = threadIdx.x; i < span.count; i += blockDim.x) {
    work.recv[span.offset + i] = received[i];
  }
  releaseRecv(recvSlot, recvTicket);
}

}  // namespace device

#undef TNCCL_HOST_DEVICE_INLINE

}  // namespace tnccl
