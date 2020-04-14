#include <cassert>
#include <iostream>
#include "Bucket.h"

using namespace std;

Bucket::Bucket()
:isInit(-1)
,arr(BUCKETSIZE)
{
  //cerr << "BUCKETSIZE=" << BUCKETSIZE << endl;
}


Bucket::~Bucket()
{
}


int Bucket::getTotalCounts()
{
    return _counts;
}


int Bucket::getSize()
{
    return _counts;
}


int Bucket::add(int id) {

    //FIFO
    if (FIFO) {
        isInit += 1;
        int index = _counts & (BUCKETSIZE - 1);
        _counts++;
        assert(index >= 0 && index < arr.size() || !(cerr << "BUCKETSIZE=" << BUCKETSIZE << " arr.size()=" << arr.size() << " index=" << index << endl));
        arr.at(index) = id;
        return index;
    }
    //Reservoir Sampling
    else {
        _counts++;
        if (index == BUCKETSIZE) {
            int randnum = rand() % (_counts) + 1;
            if (randnum == 2) {
                int randind = rand() % BUCKETSIZE;
                assert(randind >= 0 && randind < arr.size());
                arr.at(randind) = id;
                return randind;
            } else {
                return -1;
            }
        } else {
          assert(index >= 0 && index < arr.size());
          arr.at(index) = id;
            int returnIndex = index;
            index++;
            return returnIndex;
        }
    }
}


int Bucket::retrieve(int indice)
{
    if (indice >= BUCKETSIZE)
        return -1;
    assert(indice >= 0 && indice < arr.size() || !(cerr << "BUCKETSIZE=" << BUCKETSIZE << " arr.size()=" << arr.size() << " index=" << index << endl));
    return arr.at(indice);
}


int * Bucket::getAll()
{
    if (isInit == -1)
        return NULL;
    if(_counts<BUCKETSIZE){
        assert(_counts >= 0 && _counts < arr.size() || !(cerr << "BUCKETSIZE=" << BUCKETSIZE << " arr.size()=" << arr.size() << " index=" << index << endl));
        arr.at(_counts)=-1;
    }
    return arr.data();
}
