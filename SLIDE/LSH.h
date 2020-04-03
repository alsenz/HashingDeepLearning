#pragma once
#include "Bucket.h"
#include <random>

class LSH {
private:
	Bucket ** _bucket;
	int _K;
	int _L;
	int _RangePow;
	int *rand1;


public:
	LSH(int K, int L, int RangePow);
	void clear();
  std::vector<int> add(const std::vector<int> &indices, int id) const;
	int add(int indices, int tableId, int id);
  std::vector<int> hashesToIndex(const int * hashes) const;
  const int** retrieveRaw(const std::vector<int> &indices) const;
	int retrieve(int table, int indices, int bucket) const;
	void count();
	~LSH();
};
