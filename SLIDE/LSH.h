#pragma once
#include "Bucket.h"
#include "Util.h"
#include <random>
#include <vector>

class LSH {
private:
  Vec2d<Bucket> _bucket;
  int _K;
  int _L;
  int _RangePow;
  std::vector<int> _rand1;

public:
  LSH(int K, int L, int RangePow);
  void clear();
  void add(const std::vector<int> &indices, int id, bool unlimited);
  void add(int indices, int tableId, int id, bool unlimited);
  std::vector<int> hashesToIndex(const std::vector<int> &hashes) const;
  std::vector<const std::vector<int> *>
  retrieveRaw(const std::vector<int> &indices) const;
  ~LSH();
};
