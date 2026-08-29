#pragma once

#include "tnccl.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tnccl {

constexpr std::uint32_t kSlots = 4;
constexpr std::uint32_t kChunkElems = 16 * 1024;
constexpr int kKernelThreads = 256;

struct alignas(128) Slot {
  unsigned long long ready;
  unsigned long long credit;
  alignas(16) float payload[kChunkElems];
};

struct Mailbox {
  Slot slots[kSlots];
};

struct DevComm {
  int rank;
  int nranks;
  Mailbox* inbox;
  Mailbox* outbox;
};

struct DevWork {
  DevComm comm;
  const float* send;
  float* recv;
  std::uint64_t count;
  std::uint64_t baseTicket;
  std::uint64_t tilesPerSegment;
};

struct BootstrapState {
  int rank = -1;
  int nranks = 0;
  int listenFd = -1;
  int peerFd = -1;
  std::vector<int> rankFds;
  char socketPath[108] = {};
  bool active = false;
};

struct PeerMeta {
  int status;
  int device;
  cudaIpcMemHandle_t mailboxHandle;
};

tncclResult bootstrapOpen(
    BootstrapState* state, const tncclUniqueId& id, int nranks, int rank);
tncclResult bootstrapAllGather(
    BootstrapState* state, const void* send, void* recv, std::size_t bytes);
tncclResult bootstrapBarrier(BootstrapState* state);
void bootstrapClose(BootstrapState* state);

tncclResult launchRingAllReduce(const DevWork& work, cudaStream_t stream);

}  // namespace tnccl

struct tncclComm {
  int rank = -1;
  int nranks = 0;
  int device = -1;
  tnccl::Mailbox* localInbox = nullptr;
  tnccl::Mailbox* remoteNextInbox = nullptr;
  tnccl::DevComm dev{};
  std::uint64_t nextTicket = 0;
  cudaStream_t boundStream = nullptr;
  bool streamBound = false;
  tnccl::BootstrapState bootstrap;
};
