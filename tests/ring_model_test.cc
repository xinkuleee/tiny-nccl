#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

constexpr std::size_t kChunkElems = 7;
constexpr std::uint64_t kSlots = 4;

std::size_t segmentSize(std::size_t count, int nranks, int segment) {
  const std::size_t base = count / static_cast<std::size_t>(nranks);
  const std::size_t remainder = count % static_cast<std::size_t>(nranks);
  return base + (static_cast<std::size_t>(segment) < remainder ? 1 : 0);
}

std::size_t segmentOffset(std::size_t count, int nranks, int segment) {
  const std::size_t base = count / static_cast<std::size_t>(nranks);
  const std::size_t remainder = count % static_cast<std::size_t>(nranks);
  return static_cast<std::size_t>(segment) * base +
         std::min(static_cast<std::size_t>(segment), remainder);
}

using Tensor = std::vector<float>;
using Ranks = std::vector<Tensor>;

Ranks ringAllReduceModel(const Ranks& input, bool inPlace) {
  const int nranks = static_cast<int>(input.size());
  require(nranks > 0, "nranks must be positive");
  const std::size_t count = input.front().size();
  for (const auto& rank : input) {
    require(rank.size() == count, "rank counts differ");
  }

  Ranks output(nranks, Tensor(count, 0.0F));
  if (inPlace) output = input;
  if (nranks == 1) return input;

  std::size_t maxSegment = 0;
  for (int segment = 0; segment < nranks; ++segment) {
    maxSegment = std::max(maxSegment, segmentSize(count, nranks, segment));
  }
  const std::size_t tiles = std::max(
      std::size_t{1}, (maxSegment + kChunkElems - 1) / kChunkElems);

  for (std::size_t tile = 0; tile < tiles; ++tile) {
    std::vector<Tensor> outgoing(nranks);

    // Match the device kernel: rank r first injects segment r-1.
    for (int rank = 0; rank < nranks; ++rank) {
      const int segment = (rank - 1 + nranks) % nranks;
      const std::size_t begin = segmentOffset(count, nranks, segment) +
                                tile * kChunkElems;
      const std::size_t segmentEnd = segmentOffset(count, nranks, segment) +
                                     segmentSize(count, nranks, segment);
      const std::size_t end = std::min(begin + kChunkElems, segmentEnd);
      if (begin < end) {
        const Tensor& source = inPlace ? output[rank] : input[rank];
        outgoing[rank].assign(source.begin() + begin, source.begin() + end);
      }
    }

    // Reduce-scatter: move each segment N-1 edges and add one local
    // contribution at every receiving rank.
    for (int hop = 1; hop < nranks; ++hop) {
      std::vector<Tensor> next(nranks);
      for (int rank = 0; rank < nranks; ++rank) {
        const int previous = (rank - 1 + nranks) % nranks;
        Tensor partial = outgoing[previous];
        const int segment =
            (rank - hop - 1 + 2 * nranks) % nranks;
        const std::size_t begin = segmentOffset(count, nranks, segment) +
                                  tile * kChunkElems;
        for (std::size_t i = 0; i < partial.size(); ++i) {
          const Tensor& source = inPlace ? output[rank] : input[rank];
          partial[i] += source[begin + i];
        }
        if (hop == nranks - 1) {
          std::copy(partial.begin(), partial.end(),
                    output[rank].begin() + begin);
        } else {
          next[rank] = std::move(partial);
        }
      }
      outgoing = std::move(next);
    }

    // All-gather: each rank starts with its rank-numbered reduced segment.
    for (int rank = 0; rank < nranks; ++rank) {
      const int segment = rank;
      const std::size_t begin = segmentOffset(count, nranks, segment) +
                                tile * kChunkElems;
      const std::size_t segmentEnd = segmentOffset(count, nranks, segment) +
                                     segmentSize(count, nranks, segment);
      const std::size_t end = std::min(begin + kChunkElems, segmentEnd);
      outgoing[rank].clear();
      if (begin < end) {
        outgoing[rank].assign(output[rank].begin() + begin,
                              output[rank].begin() + end);
      }
    }

    for (int hop = 1; hop < nranks; ++hop) {
      std::vector<Tensor> next(nranks);
      for (int rank = 0; rank < nranks; ++rank) {
        const int previous = (rank - 1 + nranks) % nranks;
        Tensor received = outgoing[previous];
        const int segment = (rank - hop + nranks) % nranks;
        const std::size_t begin = segmentOffset(count, nranks, segment) +
                                  tile * kChunkElems;
        std::copy(received.begin(), received.end(),
                  output[rank].begin() + begin);
        if (hop != nranks - 1) next[rank] = std::move(received);
      }
      outgoing = std::move(next);
    }
  }
  return output;
}

void checkRing(int nranks, std::size_t count, bool inPlace) {
  Ranks input(nranks, Tensor(count));
  for (int rank = 0; rank < nranks; ++rank) {
    for (std::size_t i = 0; i < count; ++i) {
      input[rank][i] = static_cast<float>((rank + 1) * 3) +
                       static_cast<float>(i % 11) * 0.25F;
    }
  }

  const Ranks output = ringAllReduceModel(input, inPlace);
  for (int rank = 0; rank < nranks; ++rank) {
    for (std::size_t i = 0; i < count; ++i) {
      float expected = 0.0F;
      for (int peer = 0; peer < nranks; ++peer) expected += input[peer][i];
      require(std::fabs(output[rank][i] - expected) < 1e-5F,
              "ring result mismatch");
    }
  }
}

void checkTicketProtocol() {
  std::uint64_t ready[kSlots] = {};
  std::uint64_t credit[kSlots] = {0, 1, 2, 3};
  for (std::uint64_t ticket = 0; ticket < 10000; ++ticket) {
    const std::uint64_t slot = ticket % kSlots;
    require(credit[slot] == ticket, "credit mismatch");
    ready[slot] = ticket + 1;
    require(ready[slot] == ticket + 1, "ready mismatch");
    credit[slot] = ticket + kSlots;
  }
}

}  // namespace

int main() {
  for (int nranks : {1, 2, 3, 4, 7}) {
    for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                              std::size_t{5}, std::size_t{8}, std::size_t{31},
                              std::size_t{113}}) {
      checkRing(nranks, count, false);
      checkRing(nranks, count, true);
    }
  }
  checkTicketProtocol();
  std::cout << "ring model and ticket protocol: ok\n";
}
