#pragma once
#include "Layer.h"
#include <chrono>
#include "cnpy.h"
#include <sys/mman.h>

using namespace std;

class Network
{
private:
	Layer** _hiddenlayers;
	float _learningRate;
	int _numberOfLayers;
  const std::vector<int> &_sizesOfLayers;
	NodeType* _layersTypes;
	const std::vector<float> &_Sparsity;
	//int* _inputIDs;
	int  _currentBatchSize;


public:
	Network(const std::vector<int> &sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, const std::vector<int> &K, const std::vector<int> &L, const std::vector<int> &RangePow, const std::vector<float> &Sparsity, cnpy::npz_t arr);
	Layer* getLayer(int LayerID);
	int predictClass(const vector<int*> &inputIndices, const vector<float*> &inputValues, const vector<int> &length,  const vector<int*> &labels, const vector<int> &labelsize, int numInClass, int numOutClass) const;
	int ProcessInput(const vector<int*> &inputIndices, const vector<float*> &inputValues, const vector<int> &lengths, const vector<int*> &labels, const vector<int> &labelsize, int iter, bool rehash, bool rebuild) const;
	void saveWeights(string file);
	~Network();
	void * operator new(size_t size){
	    cout << "new Network" << endl;
	    void* ptr = mmap(NULL, size,
	        PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
	        -1, 0);
	    if (ptr == MAP_FAILED){
	        ptr = mmap(NULL, size,
	            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
	            -1, 0);
	    }
	    if (ptr == MAP_FAILED){
	        std::cout << "mmap failed at Network." << std::endl;
	    }
	    return ptr;
	}
	void operator delete(void * pointer){munmap(pointer, sizeof(Network));};
};

