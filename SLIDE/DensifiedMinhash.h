#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <vector>
#include <string.h>
#include "MurmurHash.h"
#include "Util.h"


class DensifiedMinhash
{
private:
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash;
public:
    DensifiedMinhash(int numHashes, int noOfBitsToHash);
    const int * getHash(const std::vector<int> &indices, const std::vector<float> &data, const std::vector<int> &binids, int dataLen) const;
    int getRandDoubleHash(int binid, int count) const;
    const int * getHashEasy(const std::vector<int> &binids, const std::vector<float> &data, int dataLen, int topK) const;
    const int * getHashEasy(const std::vector<int> &binids, const SubVector<float> &data, int dataLen, int topK) const;
    void getMap(int n, std::vector<int> &binid);
    ~DensifiedMinhash();
};