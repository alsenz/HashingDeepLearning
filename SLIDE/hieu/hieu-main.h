#pragma once
#include <stddef.h>
#include <vector>

namespace hieu {
int main(size_t maxBatchsize, const std::vector<int> &K, const std::vector<int> &L, const std::vector<int> &RangePow, const std::vector<float> &Sparsity);
}
