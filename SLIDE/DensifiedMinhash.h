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

class DensifiedMinhash : public HashBase {
private:
  int _randa, _numhashes, _rangePow, _lognumhash;
  std::vector<int> _randHash;
  std::vector<int> _binids;

public:
  DensifiedMinhash(int numHashes, int noOfBitsToHash);
  std::vector<int> getHash(const std::vector<int> &indices,
                           const std::vector<float> &data) const;
  int getRandDoubleHash(int binid, int count) const;
  std::vector<int> getHash(const SubVectorConst<float> &data) const override;
  void getMap(int n);

  ~DensifiedMinhash();
};