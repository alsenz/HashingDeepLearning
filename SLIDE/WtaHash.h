#pragma once
#include "HashBase.h"
#include "MurmurHash.h"
#include "Util.h"
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/*
*  Algorithm from the paper The Power of Comparative Reasoning. Jay Yagnik,
Dennis Strelow, David A. Ross, Ruei-sung Lin

*/
class WtaHash : public HashBase {
private:
  int _numhashes, _rangePow;
  std::vector<int> _indices;

public:
  WtaHash(int numHashes, int noOfBitsToHash);
  std::vector<int> getHash(const SubVectorConst<float> &data) const override;
  ~WtaHash();
};