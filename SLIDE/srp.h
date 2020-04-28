#pragma once
#include "HashBase.h"
#include "Util.h"
#include <vector>

class SparseRandomProjection : public HashBase {
private:
  size_t _dim;
  size_t _numhashes, _samSize;
  Vec2d<short> _randBits;
  Vec2d<int> _indices;

public:
  SparseRandomProjection(size_t dimention, size_t numOfHashes, int ratio);
  std::vector<int> getHash(const std::vector<float> &vector) const;
  std::vector<int> getHash(const SubVectorConst<float> &vector) const;
  std::vector<int> getHashSparse(const std::vector<int> &indices,
                                 const std::vector<float> &values) const;
  ~SparseRandomProjection();
};
