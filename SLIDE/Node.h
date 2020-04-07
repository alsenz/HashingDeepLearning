#pragma once
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <linux/mman.h>
#include <sys/mman.h>
#include <asm-generic/mman-common.h>


using namespace std;

enum NodeType
{ ReLU, Softmax};

struct train {
    float _lastDeltaforBPs;
    float _lastActivations;
    float _lastGradients;
    int _ActiveinputIds;

    void * operator new(size_t size){
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED){
            std::cout << "mmap failed at train." << std::endl;
        }
        return ptr;
    }
    void* operator new (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
    void* operator new (std::size_t size, void* ptr){return operator new (size);};
    void* operator new[] (std::size_t size){
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED){
            std::cout << "mmap fail! No train array!" << std::endl;
        }
        return ptr;
    }
    void* operator new[] (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
    void* operator new[] (std::size_t size, void* ptr){return operator new (size);};

    void operator delete(void * ptr){munmap(ptr, sizeof(train));};
    void operator delete (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(train));};
    void operator delete (void* ptr, void* voidptr2){};
    // TODO: The size to be munmap'd should be the entire array, not just a single object
    void operator delete[](void * ptr){munmap(ptr, sizeof(train));};
    void operator delete[] (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(train));};
    void operator delete[] (void* ptr, void* voidptr2){};

} __attribute__ ((aligned (64)));

class Node
{
private:
	int _activeInputs;
    NodeType _type;

    train* _train;
    int _currentBatchsize;
    size_t _dim, _layerNum, _IDinLayer;
    std::vector<int> _indicesInTables;
    std::vector<int> _indicesInBuckets;
    float* _weights;
    float* _mirrorWeights;
    float* _adamAvgMom;
    float* _adamAvgVel;
    float* _t; //for adam
    float _bias = 0;
    float _tbias = 0;
    float _adamAvgMombias = 0;
    float _adamAvgVelbias = 0;
    float _mirrorbias = 0;

public:

  Node();
  Node(const Node&) = delete;

	void Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, train* train_blob);
	float getLastActivation(int inputID) const;
	void incrementDelta(int inputID, float incrementValue) const;
	float getActivation(int* indices, float* values, int length, int inputID);
	bool getInputActive(int inputID) const;
	bool getActiveInputs(void) const;
	void SetlastActivation(int inputID, float realActivation);
	void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize) const;
	void backPropagate(const std::vector<Node> &previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
	void backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID);
	~Node();

	//only for debugging
	float purturbWeight(int weightid, float delta);
	float getGradient(int weightid, int inputID, float InputVal);

  const size_t &getDim() const 
  {
    return _dim;
  }

  void setIndicesInTables(std::vector<int> &val)
  {
    _indicesInTables = val;
  }

  void setIndicesInBuckets(const std::vector<int> &val)
  {
    _indicesInBuckets = val;
  }

  float *getWeights() const
  {
    return _weights;
  }

  float *getMirrorWeights() const
  {
    return _mirrorWeights;
  }

  float getAdamAvgMom(size_t idx) const
  {
    return _adamAvgMom[idx];
  }
  void setAdamAvgMom(size_t idx, float val)
  {
    _adamAvgMom[idx] = val;
  }

  float getAdamAvgVel(size_t idx) const
  {
    return _adamAvgVel[idx];
  }
  void setAdamAvgVel(size_t idx, float val)
  {
    _adamAvgVel[idx] = val;
  }
  
  float getT(size_t idx) const
  {
    return _t[idx];
  }
  void setT(size_t idx, float val)
  {
    _t[idx] = val;
  }

  float getBias() const
  {
    return _bias;
  }
  float &getBias()
  {
    return _bias;
  }

  float getTBias() const
  {
    return _tbias;
  }
  float &getTBias()
  {
    return _tbias;
  }

  float getAdamAvgMombias() const
  {
    return _adamAvgMombias;
  }
  float &getAdamAvgMombias()
  {
    return _adamAvgMombias;
  }

  float getAdamAvgVelbias() const
  {
    return _adamAvgVelbias;
  }
  float &getAdamAvgVelbias()
  {
    return _adamAvgVelbias;
  }

  float getMirrorBias() const
  {
    return _mirrorbias;
  }

};
