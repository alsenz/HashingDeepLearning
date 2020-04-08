#pragma once
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <linux/mman.h>
#include <sys/mman.h>
#include <asm-generic/mman-common.h>
#include "Util.h"

enum NodeType
{ ReLU, Softmax};

struct train {
    float _lastDeltaforBPs;
    float _lastActivations;
    float _lastGradients;
    int _ActiveinputIds;

} __attribute__ ((aligned (64)));

class Node
{
private:
	int _activeInputs;
    NodeType _type;

    std::vector<train> _train;
    int _currentBatchsize;
    size_t _dim, _layerNum, _IDinLayer;
    std::vector<int> _indicesInTables;
    std::vector<int> _indicesInBuckets;
    SubVector<float> *_weights;
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

	void Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, SubVector<float> *weights, float bias, float *adamAvgMom, float *adamAvgVel);
	float getLastActivation(int inputID) const;
	void incrementDelta(int inputID, float incrementValue);
	float getActivation(int* indices, float* values, int length, int inputID);
	bool getInputActive(int inputID) const;
	bool getActiveInputs(void) const;
	void SetlastActivation(int inputID, float realActivation);
	void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize);
	void backPropagate(std::vector<Node> &previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
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

  const SubVector<float> &getWeights() const
  {
    return *_weights;
  }

  SubVector<float> &getWeights()
  {
    return *_weights;
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

  const float &getTBias() const
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
