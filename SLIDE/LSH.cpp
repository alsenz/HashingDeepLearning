#include "LSH.h"
#include "Config.h"
#include <chrono>
#include <climits>
#include <iostream>
#include <unordered_map>
#include <algorithm>

using namespace std;

LSH::LSH(int K, int L, int RangePow) : _bucket(L), _seeds(L) {
  _K = K;
  _L = L;
  _RangePow = RangePow;
  _numBuckets = 1 << _RangePow;

  //#pragma omp parallel for
  for (int i = 0; i < L; i++) {
    _bucket[i].resize(_numBuckets);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, INT_MAX);

  generate(_seeds.begin(), _seeds.end(), [&]() { return dis(gen); });
}

void LSH::clear() {
  for (int i = 0; i < _L; i++) {
    _bucket[i].clear();
    _bucket[i].resize(_numBuckets);
  }
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::vector<int> LSH::hashesToIndex(const std::vector<int> &hashes) const {
  //cerr << "hashes=" << hashes.size() << " " << _K << " " << _L << endl;
  assert(hashes.size() == _K * _L);
  std::vector<int> indices(_L);
  for (int i = 0; i < _L; i++) {
    size_t index = _seeds[i];
    for (int j = 0; j < _K; j++) {
      int h = hashes[_K * i + j];
      hash_combine(index, h);
    }
    index = index % _numBuckets;
    indices[i] = index;
  }
  return indices;
}

void LSH::Add(const std::vector<int> &indices, int id, bool unlimited) {
  for (int i = 0; i < _L; i++) {
    Add(i, indices[i], id, unlimited);
  }
}

void LSH::Add(int tableId, int indices, int id, bool unlimited) {
  std::vector<Bucket> &buckets = _bucket.at(tableId);
  Bucket &bucket = buckets.at(indices);
  bucket.add(id, unlimited);
}

/*
 * Returns all the buckets
 */
std::vector<const std::vector<int> *>
LSH::retrieveRaw(const std::vector<int> &indices) const {
  std::vector<const std::vector<int> *> rawResults(_L);

  for (int i = 0; i < _L; i++) {
    rawResults[i] = &_bucket[i][indices[i]].getAll();
  }
  return rawResults;
}

LSH::~LSH() {}
