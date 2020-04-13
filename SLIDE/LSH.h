#pragma once
#include "Bucket.h"
#include <random>

class LSH {
private:
	Bucket ** _bucket;
	int _K;
	int _L;
	int _RangePow;
	std::vector<int> _rand1;


public:
	LSH(int K, int L, int RangePow);
	void clear();
  std::vector<int> add(const std::vector<int> &indices, int id);
	int add(int indices, int tableId, int id);
  std::vector<int> hashesToIndex(const std::vector<int> &hashes);
	int** retrieveRaw(const std::vector<int> &indices);
	int retrieve(int table, int indices, int bucket);
	void count();
	~LSH();
};
