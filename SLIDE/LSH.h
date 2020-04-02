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
	const int* add(const int *indices, int id);
	int add(int indices, int tableId, int id);
	const int * hashesToIndex(const int * hashes) const;
  const int** retrieveRaw(const int *indices) const;
	int retrieve(int table, int indices, int bucket) const;
	void count();
	~LSH();
};
