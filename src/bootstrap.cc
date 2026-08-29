#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal.h"

#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <poll.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>

namespace {

constexpr std::uint64_t kIdMagic = 0x544e43434c494430ull;  // TNCCLID0
constexpr std::uint64_t kWireMagic = 0x544e43434c425330ull;  // TNCCLBS0
constexpr std::uint32_t kVersion = 1;
constexpr int kTimeoutMs = 30000;
constexpr std::size_t kMaxGatherBytesPerRank = 1u << 20;

enum class MessageType : std::uint32_t {
  kHello = 1,
  kHelloAck = 2,
  kGather = 3,
  kGatherResult = 4,
  kBarrier = 5,
  kBarrierRelease = 6,
};

struct UniqueIdPayload {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t pathLength;
  char path[108];
  std::uint32_t reserved;
};

static_assert(sizeof(UniqueIdPayload) == sizeof(tncclUniqueId));

struct WireHeader {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t type;
  std::uint64_t bytes;
};

struct Hello {
  std::int32_t rank;
  std::int32_t nranks;
};

struct HelloAck {
  std::int32_t status;
};

using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;

Deadline makeDeadline() {
  return Clock::now() + std::chrono::milliseconds(kTimeoutMs);
}

int remainingMilliseconds(Deadline deadline) {
  const auto now = Clock::now();
  if (now >= deadline) return 0;
  const auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
          .count();
  return static_cast<int>(std::min<std::int64_t>(remaining + 1, INT_MAX));
}

bool waitForFd(int fd, short events, Deadline deadline) {
  while (true) {
    const int timeout = remainingMilliseconds(deadline);
    if (timeout == 0) {
      errno = ETIMEDOUT;
      return false;
    }

    pollfd descriptor{fd, events, 0};
    const int rc = poll(&descriptor, 1, timeout);
    if (rc > 0) {
      if (descriptor.revents & POLLNVAL) {
        errno = EBADF;
        return false;
      }
      if (descriptor.revents & events) return true;
      if (descriptor.revents & (POLLERR | POLLHUP)) {
        errno = ECONNRESET;
        return false;
      }
      continue;
    }
    if (rc == 0) {
      errno = ETIMEDOUT;
      return false;
    }
    if (errno != EINTR) return false;
  }
}

bool readFull(int fd, void* buffer, std::size_t bytes, Deadline deadline) {
  auto* cursor = static_cast<unsigned char*>(buffer);
  std::size_t done = 0;
  while (done != bytes) {
    const ssize_t rc = recv(fd, cursor + done, bytes - done, 0);
    if (rc > 0) {
      done += static_cast<std::size_t>(rc);
      continue;
    }
    if (rc == 0) {
      errno = ECONNRESET;
      return false;
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      if (!waitForFd(fd, POLLIN, deadline)) return false;
      continue;
    }
    return false;
  }
  return true;
}

bool writeFull(int fd, const void* buffer, std::size_t bytes,
               Deadline deadline) {
  const auto* cursor = static_cast<const unsigned char*>(buffer);
  std::size_t done = 0;
  while (done != bytes) {
    const ssize_t rc =
        send(fd, cursor + done, bytes - done, MSG_NOSIGNAL);
    if (rc > 0) {
      done += static_cast<std::size_t>(rc);
      continue;
    }
    if (rc == 0) {
      errno = EPIPE;
      return false;
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      if (!waitForFd(fd, POLLOUT, deadline)) return false;
      continue;
    }
    return false;
  }
  return true;
}

WireHeader makeHeader(MessageType type, std::uint64_t bytes) {
  return WireHeader{kWireMagic, kVersion, static_cast<std::uint32_t>(type),
                    bytes};
}

bool validHeader(const WireHeader& header, MessageType type,
                 std::uint64_t bytes) {
  return header.magic == kWireMagic && header.version == kVersion &&
         header.type == static_cast<std::uint32_t>(type) &&
         header.bytes == bytes;
}

bool sendMessage(int fd, MessageType type, const void* payload,
                 std::size_t bytes, Deadline deadline) {
  const WireHeader header = makeHeader(type, bytes);
  return writeFull(fd, &header, sizeof(header), deadline) &&
         (bytes == 0 || writeFull(fd, payload, bytes, deadline));
}

bool receiveMessage(int fd, MessageType type, void* payload,
                    std::size_t bytes, Deadline deadline) {
  WireHeader header{};
  return readFull(fd, &header, sizeof(header), deadline) &&
         validHeader(header, type, bytes) &&
         (bytes == 0 || readFull(fd, payload, bytes, deadline));
}

bool fillRandom(void* buffer, std::size_t bytes) {
  auto* cursor = static_cast<unsigned char*>(buffer);
  std::size_t done = 0;
  while (done != bytes) {
    const ssize_t rc = getrandom(cursor + done, bytes - done, 0);
    if (rc > 0) {
      done += static_cast<std::size_t>(rc);
      continue;
    }
    if (rc < 0 && errno == EINTR) continue;
    break;
  }
  if (done == bytes) return true;

  const int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  while (done != bytes) {
    const ssize_t rc = read(fd, cursor + done, bytes - done);
    if (rc > 0) {
      done += static_cast<std::size_t>(rc);
      continue;
    }
    if (rc < 0 && errno == EINTR) continue;
    close(fd);
    return false;
  }
  close(fd);
  return true;
}

bool decodeSocketPath(const tncclUniqueId& id, char (&path)[108]) {
  UniqueIdPayload payload{};
  std::memcpy(&payload, id.bytes, sizeof(payload));
  constexpr char kPrefix[] = "/tmp/tnccl-";
  constexpr std::size_t kPrefixLength = sizeof(kPrefix) - 1;

  if (payload.magic != kIdMagic || payload.version != kVersion ||
      payload.pathLength < kPrefixLength ||
      payload.pathLength >= sizeof(payload.path) ||
      std::memcmp(payload.path, kPrefix, kPrefixLength) != 0 ||
      std::memchr(payload.path, '\0', payload.pathLength) != nullptr) {
    return false;
  }
  for (std::size_t i = kPrefixLength; i < payload.pathLength; ++i) {
    const unsigned char ch = static_cast<unsigned char>(payload.path[i]);
    const bool allowed = (ch >= 'a' && ch <= 'z') ||
                         (ch >= 'A' && ch <= 'Z') ||
                         (ch >= '0' && ch <= '9') || ch == '-' ||
                         ch == '_' || ch == '.';
    if (!allowed) return false;
  }

  std::memcpy(path, payload.path, payload.pathLength);
  path[payload.pathLength] = '\0';
  return true;
}

sockaddr_un makeAddress(const char* path) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, path, sizeof(address.sun_path) - 1);
  return address;
}

socklen_t addressLength(const sockaddr_un& address) {
  return static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) +
                                std::strlen(address.sun_path) + 1);
}

int createSocket() {
  return socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
}

int connectToRankZero(const char* path, Deadline deadline) {
  const sockaddr_un address = makeAddress(path);
  while (remainingMilliseconds(deadline) != 0) {
    const int fd = createSocket();
    if (fd < 0) return -1;

    if (connect(fd, reinterpret_cast<const sockaddr*>(&address),
                addressLength(address)) == 0) {
      return fd;
    }

    int connectError = errno;
    if (connectError == EINPROGRESS) {
      if (waitForFd(fd, POLLOUT, deadline)) {
        socklen_t length = sizeof(connectError);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &connectError, &length) == 0 &&
            connectError == 0) {
          return fd;
        }
      } else {
        connectError = errno;
      }
    }

    close(fd);
    if (connectError != ENOENT && connectError != ECONNREFUSED &&
        connectError != EINTR) {
      errno = connectError;
      return -1;
    }

    const int pauseMs = std::min(10, remainingMilliseconds(deadline));
    if (pauseMs > 0) poll(nullptr, 0, pauseMs);
  }
  errno = ETIMEDOUT;
  return -1;
}

int acceptPeer(int listenFd, Deadline deadline) {
  while (true) {
    const int fd =
        accept4(listenFd, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd >= 0) return fd;
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      if (!waitForFd(listenFd, POLLIN, deadline)) return -1;
      continue;
    }
    return -1;
  }
}

tncclResult decodeStatus(std::int32_t value) {
  if (value < TNCCL_SUCCESS || value > TNCCL_CUDA_ERROR) {
    return TNCCL_SYSTEM_ERROR;
  }
  return static_cast<tncclResult>(value);
}

}  // namespace

extern "C" tncclResult tncclGetUniqueId(tncclUniqueId* id) {
  if (id == nullptr) return TNCCL_INVALID_ARGUMENT;

  unsigned char random[16]{};
  if (!fillRandom(random, sizeof(random))) return TNCCL_SYSTEM_ERROR;

  char hex[sizeof(random) * 2 + 1]{};
  constexpr char kDigits[] = "0123456789abcdef";
  for (std::size_t i = 0; i < sizeof(random); ++i) {
    hex[2 * i] = kDigits[random[i] >> 4];
    hex[2 * i + 1] = kDigits[random[i] & 0x0f];
  }

  UniqueIdPayload payload{};
  payload.magic = kIdMagic;
  payload.version = kVersion;
  const int written = std::snprintf(
      payload.path, sizeof(payload.path), "/tmp/tnccl-%lu-%ld-%s.sock",
      static_cast<unsigned long>(geteuid()), static_cast<long>(getpid()), hex);
  if (written <= 0 || static_cast<std::size_t>(written) >= sizeof(payload.path)) {
    return TNCCL_SYSTEM_ERROR;
  }
  payload.pathLength = static_cast<std::uint32_t>(written);
  std::memcpy(id->bytes, &payload, sizeof(payload));
  return TNCCL_SUCCESS;
}

namespace tnccl {

tncclResult bootstrapOpen(BootstrapState* state, const tncclUniqueId& id,
                          int nranks, int rank) {
  if (state == nullptr || state->active || nranks <= 1 || rank < 0 ||
      rank >= nranks) {
    return TNCCL_INVALID_ARGUMENT;
  }

  char path[108]{};
  if (!decodeSocketPath(id, path)) return TNCCL_INVALID_ARGUMENT;

  state->rank = rank;
  state->nranks = nranks;
  const Deadline deadline = makeDeadline();

  if (rank == 0) {
    try {
      state->rankFds.assign(static_cast<std::size_t>(nranks), -1);
    } catch (const std::bad_alloc&) {
      bootstrapClose(state);
      return TNCCL_SYSTEM_ERROR;
    }
    state->listenFd = createSocket();
    if (state->listenFd < 0) {
      bootstrapClose(state);
      return TNCCL_SYSTEM_ERROR;
    }
    state->active = true;

    const sockaddr_un address = makeAddress(path);
    if (bind(state->listenFd, reinterpret_cast<const sockaddr*>(&address),
             addressLength(address)) != 0) {
      // The path may belong to another live rendezvous. Do not unlink it when
      // this rank never acquired the pathname.
      state->socketPath[0] = '\0';
      bootstrapClose(state);
      return TNCCL_SYSTEM_ERROR;
    }
    std::strncpy(state->socketPath, path, sizeof(state->socketPath) - 1);
    if (chmod(path, S_IRUSR | S_IWUSR) != 0 ||
        listen(state->listenFd, nranks - 1) != 0) {
      bootstrapClose(state);
      return TNCCL_SYSTEM_ERROR;
    }

    for (int connected = 1; connected < nranks; ++connected) {
      const int fd = acceptPeer(state->listenFd, deadline);
      if (fd < 0) {
        bootstrapClose(state);
        return TNCCL_SYSTEM_ERROR;
      }

      Hello hello{};
      if (!receiveMessage(fd, MessageType::kHello, &hello, sizeof(hello),
                          deadline)) {
        close(fd);
        bootstrapClose(state);
        return TNCCL_SYSTEM_ERROR;
      }

      tncclResult status = TNCCL_SUCCESS;
      if (hello.nranks != nranks || hello.rank <= 0 ||
          hello.rank >= nranks || state->rankFds[hello.rank] != -1) {
        status = TNCCL_INVALID_ARGUMENT;
      }
      const HelloAck ack{static_cast<std::int32_t>(status)};
      if (!sendMessage(fd, MessageType::kHelloAck, &ack, sizeof(ack),
                       deadline)) {
        close(fd);
        bootstrapClose(state);
        return TNCCL_SYSTEM_ERROR;
      }
      if (status != TNCCL_SUCCESS) {
        close(fd);
        bootstrapClose(state);
        return status;
      }
      state->rankFds[hello.rank] = fd;
    }
    close(state->listenFd);
    state->listenFd = -1;
    return TNCCL_SUCCESS;
  }

  std::strncpy(state->socketPath, path, sizeof(state->socketPath) - 1);
  state->peerFd = connectToRankZero(path, deadline);
  if (state->peerFd < 0) {
    bootstrapClose(state);
    return TNCCL_SYSTEM_ERROR;
  }
  state->active = true;

  const Hello hello{rank, nranks};
  HelloAck ack{};
  if (!sendMessage(state->peerFd, MessageType::kHello, &hello, sizeof(hello),
                   deadline) ||
      !receiveMessage(state->peerFd, MessageType::kHelloAck, &ack, sizeof(ack),
                      deadline)) {
    bootstrapClose(state);
    return TNCCL_SYSTEM_ERROR;
  }
  const tncclResult status = decodeStatus(ack.status);
  if (status != TNCCL_SUCCESS) bootstrapClose(state);
  return status;
}

tncclResult bootstrapAllGather(BootstrapState* state, const void* send,
                               void* recv, std::size_t bytes) {
  if (state == nullptr || !state->active || state->nranks <= 1 ||
      (bytes != 0 && (send == nullptr || recv == nullptr)) ||
      bytes > kMaxGatherBytesPerRank ||
      bytes > std::numeric_limits<std::size_t>::max() /
                  static_cast<std::size_t>(state->nranks)) {
    return TNCCL_INVALID_ARGUMENT;
  }

  const std::size_t total = bytes * static_cast<std::size_t>(state->nranks);
  const Deadline deadline = makeDeadline();
  if (state->rank == 0) {
    auto* output = static_cast<unsigned char*>(recv);
    if (bytes != 0) std::memcpy(output, send, bytes);

    for (int peer = 1; peer < state->nranks; ++peer) {
      void* destination =
          bytes == 0
              ? nullptr
              : output + static_cast<std::size_t>(peer) * bytes;
      if (!receiveMessage(state->rankFds[peer], MessageType::kGather,
                          destination, bytes, deadline)) {
        return TNCCL_SYSTEM_ERROR;
      }
    }
    for (int peer = 1; peer < state->nranks; ++peer) {
      if (!sendMessage(state->rankFds[peer], MessageType::kGatherResult, recv,
                       total, deadline)) {
        return TNCCL_SYSTEM_ERROR;
      }
    }
    return TNCCL_SUCCESS;
  }

  if (!sendMessage(state->peerFd, MessageType::kGather, send, bytes, deadline) ||
      !receiveMessage(state->peerFd, MessageType::kGatherResult, recv, total,
                      deadline)) {
    return TNCCL_SYSTEM_ERROR;
  }
  return TNCCL_SUCCESS;
}

tncclResult bootstrapBarrier(BootstrapState* state) {
  if (state == nullptr || !state->active || state->nranks <= 1) {
    return TNCCL_INVALID_ARGUMENT;
  }

  const Deadline deadline = makeDeadline();
  if (state->rank == 0) {
    for (int peer = 1; peer < state->nranks; ++peer) {
      if (!receiveMessage(state->rankFds[peer], MessageType::kBarrier, nullptr,
                          0, deadline)) {
        return TNCCL_SYSTEM_ERROR;
      }
    }
    for (int peer = 1; peer < state->nranks; ++peer) {
      if (!sendMessage(state->rankFds[peer], MessageType::kBarrierRelease,
                       nullptr, 0, deadline)) {
        return TNCCL_SYSTEM_ERROR;
      }
    }
    return TNCCL_SUCCESS;
  }

  if (!sendMessage(state->peerFd, MessageType::kBarrier, nullptr, 0, deadline) ||
      !receiveMessage(state->peerFd, MessageType::kBarrierRelease, nullptr, 0,
                      deadline)) {
    return TNCCL_SYSTEM_ERROR;
  }
  return TNCCL_SUCCESS;
}

void bootstrapClose(BootstrapState* state) {
  if (state == nullptr) return;

  if (state->peerFd >= 0) {
    close(state->peerFd);
    state->peerFd = -1;
  }
  for (int& fd : state->rankFds) {
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  }
  state->rankFds.clear();
  if (state->listenFd >= 0) {
    close(state->listenFd);
    state->listenFd = -1;
  }
  if (state->rank == 0 && state->socketPath[0] != '\0') {
    unlink(state->socketPath);
  }
  state->rank = -1;
  state->nranks = 0;
  state->socketPath[0] = '\0';
  state->active = false;
}

}  // namespace tnccl
