#include "tnccl.h"

#include <cuda_runtime_api.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool cudaOk(cudaError_t status, const char* operation, int rank) {
  if (status == cudaSuccess) return true;
  std::cerr << "rank " << rank << ": " << operation << ": "
            << cudaGetErrorString(status) << '\n';
  return false;
}

int runRank(int rank, int nranks, std::size_t count, tncclUniqueId id) {
  if (!cudaOk(cudaSetDevice(rank), "cudaSetDevice", rank)) return 1;

  tncclComm_t comm = nullptr;
  tncclResult result = tncclCommInitRank(&comm, nranks, id, rank);
  if (result != TNCCL_SUCCESS) {
    std::cerr << "rank " << rank << ": init: "
              << tncclGetErrorString(result) << '\n';
    return 1;
  }

  cudaStream_t stream = nullptr;
  float* send = nullptr;
  float* recv = nullptr;
  std::vector<float> hostSend(count, static_cast<float>(rank + 1));
  std::vector<float> hostRecv(count);
  bool ok = cudaOk(cudaStreamCreate(&stream), "cudaStreamCreate", rank);
  if (ok && count != 0) {
    ok = cudaOk(cudaMalloc(reinterpret_cast<void**>(&send),
                           count * sizeof(float)),
                "cudaMalloc(send)", rank) &&
         cudaOk(cudaMalloc(reinterpret_cast<void**>(&recv),
                           count * sizeof(float)),
                "cudaMalloc(recv)", rank) &&
         cudaOk(cudaMemcpyAsync(send, hostSend.data(), count * sizeof(float),
                                cudaMemcpyHostToDevice, stream),
                "copy input", rank);
  }

  if (ok) {
    result = tncclAllReduceSumF32(send, recv, count, comm, stream);
    ok = result == TNCCL_SUCCESS;
    if (!ok) {
      std::cerr << "rank " << rank << ": all-reduce: "
                << tncclGetErrorString(result) << '\n';
    }
  }
  if (ok && count != 0) {
    ok = cudaOk(cudaMemcpyAsync(hostRecv.data(), recv, count * sizeof(float),
                                cudaMemcpyDeviceToHost, stream),
                "copy output", rank) &&
         cudaOk(cudaStreamSynchronize(stream), "stream synchronize", rank);
  } else if (ok) {
    ok = cudaOk(cudaStreamSynchronize(stream), "stream synchronize", rank);
  }

  const float expected =
      static_cast<float>(nranks * (nranks + 1) / 2);
  if (ok) {
    for (std::size_t i = 0; i < count; ++i) {
      if (std::fabs(hostRecv[i] - expected) > 1e-5F) {
        std::cerr << "rank " << rank << ": mismatch at " << i << ": "
                  << hostRecv[i] << " != " << expected << '\n';
        ok = false;
        break;
      }
    }
  }

  const tncclResult destroyResult = tncclCommDestroy(comm);
  if (destroyResult != TNCCL_SUCCESS) ok = false;
  if (recv != nullptr) cudaFree(recv);
  if (send != nullptr) cudaFree(send);
  if (stream != nullptr) cudaStreamDestroy(stream);
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <nranks> <count>\n";
    return 2;
  }
  const int nranks = std::atoi(argv[1]);
  const std::size_t count = std::strtoull(argv[2], nullptr, 10);
  if (nranks < 1) {
    std::cerr << "nranks must be positive\n";
    return 2;
  }

  tncclUniqueId id{};
  if (tncclGetUniqueId(&id) != TNCCL_SUCCESS) return 1;

  std::vector<pid_t> children;
  for (int rank = 0; rank < nranks; ++rank) {
    const pid_t pid = fork();
    if (pid == 0) _exit(runRank(rank, nranks, count, id));
    if (pid < 0) {
      std::perror("fork");
      return 1;
    }
    children.push_back(pid);
  }

  bool ok = true;
  for (pid_t child : children) {
    int status = 0;
    if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0) {
      ok = false;
    }
  }
  if (ok) std::cout << "all ranks: ok\n";
  return ok ? 0 : 1;
}
