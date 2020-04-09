#pragma once
#include  <cstddef>
#include <vector>
#include "Util.h"

class SparseRandomProjection 
{
private:
	size_t _dim;
	size_t _numhashes, _samSize;
	short ** _randBits;
	int ** _indices;
public:
	SparseRandomProjection(size_t dimention, size_t numOfHashes, int ratio);
  const int * getHash(const std::vector<float> &vector, int length) const;
  const int * getHash(const SubVector<float> &vector, int length) const;
  const int * getHashSparse(std::vector<int> &indices, std::vector<float> &values, size_t length) const;
  const int * getHashSparse(int* indices, float *values, size_t length) const;
	~SparseRandomProjection();
};
