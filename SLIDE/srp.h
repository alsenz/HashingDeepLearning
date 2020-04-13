#pragma once
#include <vector>
#include "Util.h"

using namespace std;

class SparseRandomProjection 
{
private:
	size_t _dim;
	size_t _numhashes, _samSize;
	short ** _randBits;
	int ** _indices;
public:
	SparseRandomProjection(size_t dimention, size_t numOfHashes, int ratio);
	int * getHash(const std::vector<float> &vector, int length);
  int * getHash(const SubVectorConst<float> &vector, int length);
  int * getHashSparse(const std::vector<int> &indices, const std::vector<float> &values, size_t length);
	~SparseRandomProjection();
};
