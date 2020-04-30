#pragma once
#include "HashBase.h"
#include "Util.h"
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace slide {

  /*
   *  Algorithm from the paper Densified Winner Take All (WTA) Hashing for Sparse
   * Datasets. Beidi Chen, Anshumali Shrivastava
   */
  class DensifiedWtaHash : public HashBase {
  private:
    int _randa, _numhashes, _rangePow, _lognumhash, _permute;
    std::vector<int> _randHash, _indices, _pos;

  public:
    DensifiedWtaHash(int numHashes, int noOfBitsToHash);
    std::vector<int> getHash(const std::vector<int> &indices,
      const std::vector<float> &data) const;
    int getRandDoubleHash(int binid, int count) const;
    std::vector<int> getHash(const SubVectorConst<float> &data) const override;
    ~DensifiedWtaHash();
  };

}

