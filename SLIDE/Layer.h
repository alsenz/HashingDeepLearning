#pragma once
#include "Node.h"
#include "WtaHash.h"
#include "DensifiedMinhash.h"
#include "srp.h"
#include "LSH.h"
#include "DensifiedWtaHash.h"
#include "cnpy.h"
#include <sys/mman.h>

using namespace std;

class Layer
{
private:
	NodeType _type;
	Node* _Nodes;
  std::vector<int> _randNode;
  std::vector<float> _normalizationConstants;
    int _K, _L, _RangeRow, _previousLayerNumOfNodes, _batchsize;
    train* _train_array;


    int _layerID, _noOfActive;
    size_t _noOfNodes;
    float* _weights;
    float* _adamAvgMom;
    float* _adamAvgVel;
    float* _bias;
    LSH *_hashTables;
    std::vector<int> _binids;

    WtaHash *_wtaHasher;
    DensifiedMinhash *_MinHasher;
    SparseRandomProjection *_srp;
    DensifiedWtaHash *_dwtaHasher;

public:

  Layer(size_t _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity, float* weights=NULL, float* bias=NULL, float *adamAvgMom=NULL, float *adamAvgVel=NULL);
	Node &getNodebyID(size_t nodeID) const;
	Node* getAllNodes() const;
	int getNodeCount() const;
	void addtoHashTable(float* weights, int length, float bias, int id);
	float getNomalizationConstant(int inputID) const;
	int queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  const int* label, int labelsize, float Sparsity);
	void saveWeights(string file) const;
	void updateTable();
	void updateRandomNodes();

	~Layer();

    void * operator new(size_t size){
        cout << "new Layer" << endl;
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED)
            std::cout << "mmap fail! No new layer!" << std::endl;
        return ptr;};
    void operator delete(void * pointer){munmap(pointer, sizeof(Layer));};

    size_t getNoOfNodes() const
    {
      return _noOfNodes; 
    }

    LSH &getHashTables() const
    {
      return *_hashTables;
    }

    const std::vector<int> &getBinIds() const
    {
      return _binids;
    }

    WtaHash &getWTAHasher() const
    {
      return *_wtaHasher;
    }
    DensifiedMinhash &getDensifiedMinhash() const
    {
      return *_MinHasher;
    }
    SparseRandomProjection &getSparseRandomProjection() const
    {
      return *_srp;
    }
    DensifiedWtaHash &getDensifiedWtaHash() const
    {
      return *_dwtaHasher;
    }
};
