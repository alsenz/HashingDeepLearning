#include "LSH.h"
#include "Config.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <functional>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace slide {

LSH::LSH(int K, int L, int RangePow) : _bucket(L), _seeds(L), _rand1(K * L) {
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

  for (int i = 0; i < _K * _L; i++) {
    _rand1[i] = dis(gen);
    if (_rand1[i] % 2 == 0)
      _rand1[i]++;
  }
}

void LSH::clear() {
  for (int i = 0; i < _L; i++) {
    _bucket[i].clear();
    _bucket[i].resize(_numBuckets);
  }
}

template <class T> inline void hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::vector<size_t> LSH::hashesToIndex(const std::vector<int> &hashes) const {
  // cerr << "hashesToIndex1 hashes=" << hashes.size() << " " << _K << " " << _L
  // << endl;
  assert(hashes.size() == _K * _L);
  std::vector<size_t> indices(_L);
  for (int i = 0; i < _L; i++) {
    size_t index;
    if (HashFunction == 1 || HashFunction == 2) { // | HashFunction == 4) {
      index = 0;
    } else {
      index = _seeds[i];
    }

    for (int j = 0; j < _K; j++) {
      if (HashFunction == 1 || HashFunction == 2) {
        unsigned int h = hashes[_K * i + j];
        index += h << ((_K - 1 - j) * (int)floor(log(g_binsize)));
      }
      /* crap - worse than default
      else if (HashFunction == 4) {
        unsigned int h = hashes[_K * i + j];
        index += h << (_K - 1 - j);
      }
      */
      else if (HashFunction == 3) {
        unsigned int h = _rand1[_K * i + j];
        h *= _rand1[_K * i + j];
        h ^= h >> 13;
        h ^= _rand1[_K * i + j];
        index += h * hashes[_K * i + j];
      } else {
        int h = hashes[_K * i + j];
        size_t hash = std::hash<int>{}(h);
        hash_combine(index, hash);
      }
    }

    if (HashFunction == 3) {
      index = index & ((1 << _RangePow) - 1);
    }
    index = index % _numBuckets;
    indices[i] = index;
  }

  // cerr << "hashesToIndex2" << endl;
  return indices;
}

/*
std::vector<size_t> LSH::hashesToIndex(const std::vector<int> &hashes) const {
  std::vector<size_t> indices(_L);
  for (int i = 0; i < _L; i++) {
    unsigned int index = 0;

    for (int j = 0; j < _K; j++) {
      if (HashFunction == 4) {
        unsigned int h = hashes[_K * i + j];
        index += h << (_K - 1 - j);
      }
      else if (HashFunction == 1 | HashFunction == 2) {
        unsigned int h = hashes[_K * i + j];
        index += h << ((_K - 1 - j) * (int)floor(log(binsize)));

      }
      else {
        unsigned int h = _rand1[_K * i + j];
        h *= _rand1[_K * i + j];
        h ^= h >> 13;
        h ^= _rand1[_K * i + j];
        index += h * hashes[_K * i + j];
      }
    }
    if (HashFunction == 3) {
      index = index & ((1 << _RangePow) - 1);
    }
    indices[i] = index;
  }

  return indices;
}
*/

void LSH::Add(const std::vector<size_t> &indices, int id, bool unlimited) {
  for (int i = 0; i < _L; i++) {
    Add(i, indices[i], id, unlimited);
  }
}

void LSH::Add(size_t tableId, size_t indices, int id, bool unlimited) {
  std::vector<Bucket> &buckets = _bucket.at(tableId);
  Bucket &bucket = buckets.at(indices);
  bucket.add(id, unlimited);
}

/*
 * Returns all the buckets
 */
std::vector<const std::vector<int> *>
LSH::retrieveRaw(const std::vector<size_t> &indices) const {
  std::vector<const std::vector<int> *> rawResults(_L);

  for (int i = 0; i < _L; i++) {
    rawResults[i] = &_bucket[i][indices[i]].getAll();
  }
  return rawResults;
}

LSH::~LSH() {}

} // namespace slide
