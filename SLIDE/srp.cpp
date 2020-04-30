#include "srp.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

SparseRandomProjection::SparseRandomProjection(size_t dimension,
                                               size_t numOfHashes, int ratio) {
  _dim = dimension;
  _numhashes = numOfHashes;
  _samSize = ceil(1.0 * _dim / ratio);

  std::vector<int> a(_dim);
  for (size_t i = 0; i < _dim; i++) {
    a[i] = i;
  }

  _randBits.resize(_numhashes);
  _indices.resize(_numhashes);

  for (size_t i = 0; i < _numhashes; i++) {
    random_shuffle(a.begin(), a.end());
    _randBits[i].resize(_samSize);
    _indices[i].resize(_samSize);
    for (size_t j = 0; j < _samSize; j++) {
      _indices[i][j] = a[j];
      int curr = rand();
      if (curr % 2 == 0) {
        _randBits[i][j] = 1;
      } else {
        _randBits[i][j] = -1;
      }
    }
    std::sort(_indices[i].begin(), _indices[i].end());
  }
}

std::vector<int>
SparseRandomProjection::getHash(const SubVectorConst<float> &vector) const {
  // cerr << "getHash1 " << vector.size() << endl;
  // length should be = to _dim
  std::vector<int> hashes(_numhashes);

  // #pragma omp parallel for
  for (size_t i = 0; i < _numhashes; i++) {
    // cerr << "getHash2 " << endl;
    double s = 0;
    for (size_t j = 0; j < _samSize; j++) {
      // cerr << "getHash3 " << endl;
      int idx = _indices.at(i).at(j);
      float v = vector.at(idx);
      if (_randBits.at(i).at(j) >= 0) {
        s += v;
      } else {
        s -= v;
      }
      // cerr << "getHash4 " << endl;
    }
    // cerr << "getHash5 " << endl;
    hashes.at(i) = (s >= 0 ? 0 : 1);
  }
  // cerr << "getHash6 " << endl;
  return hashes;
}

std::vector<int>
SparseRandomProjection::getHashSparse(const std::vector<int> &indices,
                                      const std::vector<float> &values) const {
  std::vector<int> hashes(_numhashes);

  for (size_t p = 0; p < _numhashes; p++) {
    double s = 0;
    size_t i = 0;
    size_t j = 0;
    while (i < values.size() && j < _samSize) {
      if (indices[i] == _indices[p][j]) {
        float v = values[i];
        if (_randBits[p][j] >= 0) {
          s += v;
        } else {
          s -= v;
        }
        i++;
        j++;
      } else if (indices[i] < _indices[p][j]) {
        i++;
      } else {
        j++;
      }
    }
    hashes[p] = (s >= 0 ? 0 : 1);
  }

  return hashes;
}

SparseRandomProjection::~SparseRandomProjection() {}
