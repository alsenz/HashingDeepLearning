#pragma once
#include "Bucket.h"
#include "Util.h"
#include <random>
#include <vector>

namespace slide {

class LSH {
private:
  Vec2d<Bucket> _bucket;
  int _K;
  int _L;
  int _RangePow;
  size_t _numBuckets;
  std::vector<size_t> _seeds;
  std::vector<size_t> _rand1;

  void Add(size_t indices, size_t tableId, int id, bool unlimited);

public:
  LSH(int K, int L, int RangePow);
  void clear();
  void Add(const std::vector<size_t> &indices, int id, bool unlimited);
  std::vector<size_t> hashesToIndex(const std::vector<int> &hashes) const;
  std::vector<const std::vector<int> *>
  retrieveRaw(const std::vector<size_t> &indices) const;
  ~LSH();
};

} // namespace slide
