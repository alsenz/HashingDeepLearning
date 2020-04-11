#pragma once
#include "Layer.h"
#include <chrono>
#include <vector>
#include "cnpy.h"
#include <sys/mman.h>

using namespace std;

class Network
{
private:
	std::vector<Layer*> _hiddenlayers;
	const float _learningRate;
  const int _numberOfLayers;
  const std::vector<int> _sizesOfLayers;
  const std::vector<NodeType> _layersTypes;
  const std::vector<float> _Sparsity;
  const int  _currentBatchSize;


public:
	Network(const std::vector<int> &sizesOfLayers, const std::vector<NodeType> &layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, int* K, int* L, int* RangePow, const std::vector<float> &Sparsity, cnpy::npz_t arr);
	Layer* getLayer(int LayerID);
	int predictClass(int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
	int ProcessInput(int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize, int iter, bool rehash, bool rebuild);
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

