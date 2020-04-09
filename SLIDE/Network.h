#pragma once
#include "Layer.h"
#include <chrono>
#include "cnpy.h"
#include <sys/mman.h>

class Network
{
private:
	std::vector<Layer*> _hiddenlayers;
	float _learningRate;
	int _numberOfLayers;
  const std::vector<int> &_sizesOfLayers;
  const std::vector<NodeType> &_layersTypes;
	const std::vector<float> &_Sparsity;
	//int* _inputIDs;
	int  _currentBatchSize;

  Layer &getLayer(int LayerID);
  const Layer &getLayer(int LayerID) const;

public:
  Network() = delete;
  Network(const Network&) = delete;
  Network(const std::vector<int> &sizesOfLayers, const std::vector<NodeType> &layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, const std::vector<int> &K, const std::vector<int> &L, const std::vector<int> &RangePow, const std::vector<float> &Sparsity, cnpy::npz_t arr);
	int predictClass(std::vector< std::vector<int> > &inputIndices, const std::vector< std::vector<float> > &inputValues, const std::vector<int> &length,  const std::vector<int*> &labels, const std::vector<int> &labelsize);
	int ProcessInput(std::vector< std::vector<int> > &inputIndices, const std::vector< std::vector<float> > &inputValues, const std::vector<int> &lengths, const std::vector<int*> &labels, const std::vector<int> &labelsize, int iter, bool rehash, bool rebuild);
	void saveWeights(std::string file);
	~Network();
};

