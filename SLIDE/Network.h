#pragma once
#include "Layer.h"
#include <chrono>
#include "cnpy.h"
#include <sys/mman.h>

using namespace std;

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
	int predictClass(const vector<int*> &inputIndices, const vector<float*> &inputValues, const vector<int> &length,  const vector<int*> &labels, const vector<int> &labelsize, int numInClass, int numOutClass);
	int ProcessInput(const vector<int*> &inputIndices, const vector<float*> &inputValues, const vector<int> &lengths, const vector<int*> &labels, const vector<int> &labelsize, int iter, bool rehash, bool rebuild);
	void saveWeights(string file);
	~Network();
};

