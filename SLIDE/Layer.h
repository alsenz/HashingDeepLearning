#pragma once
#include "Node.h"
#include "WtaHash.h"
#include "DensifiedMinhash.h"
#include "srp.h"
#include "LSH.h"
#include "DensifiedWtaHash.h"
#include "cnpy.h"
#include <sys/mman.h>

class Layer
{
private:
	NodeType _type;
  std::vector<Node> _Nodes;
  std::vector<int> _randNode;
  std::vector<float> _normalizationConstants;
    int _K, _L, _RangeRow, _previousLayerNumOfNodes, _batchsize;


    int _layerID, _noOfActive;
    size_t _noOfNodes;
    std::vector<float> _weights;
    std::vector<float> _bias;
    std::vector<float> _adamAvgMom;
    std::vector<float> _adamAvgVel;
    LSH _hashTables;
    std::vector<int> _binids;

    WtaHash *_wtaHasher;
    DensifiedMinhash *_MinHasher;
    SparseRandomProjection *_srp;
    DensifiedWtaHash *_dwtaHasher;

public:
  Layer(const Layer&) = delete;
  Layer(size_t _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity);
	
  const Node &getNodebyID(size_t nodeID) const;
  Node &getNodebyID(size_t nodeID);

	std::vector<Node> &getAllNodes();
	int getNodeCount() const;
	void addtoHashTable(float* weights, int length, float bias, int id);
	float getNomalizationConstant(int inputID) const;
	int queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  const int* label, int labelsize, float Sparsity);
	void saveWeights(const std::string &file) const;
	void updateTable();
	void updateRandomNodes();

	~Layer();

    size_t getNoOfNodes() const
    {
      return _noOfNodes; 
    }

    LSH &getHashTables()
    {
      return _hashTables;
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
