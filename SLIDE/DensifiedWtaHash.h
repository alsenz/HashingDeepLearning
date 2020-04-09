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
/*
*  Algorithm from the paper Densified Winner Take All (WTA) Hashing for Sparse Datasets. Beidi Chen, Anshumali Shrivastava
*/
class DensifiedWtaHash
{
private:
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash, *_indices, *_pos, _permute;
public:
    DensifiedWtaHash(int numHashes, int noOfBitsToHash);
    const int * getHash(std::vector<int> &indices, float* data, int dataLen) const;
    const int * getHash(int* indices, float* data, int dataLen) const;
    const int getRandDoubleHash(int binid, int count) const;
    const int * getHashEasy(const float* data, int dataLen, int topK) const;
    ~DensifiedWtaHash();
};