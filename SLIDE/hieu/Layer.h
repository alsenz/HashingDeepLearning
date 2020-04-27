#pragma once
#include "../DensifiedWtaHash.h"
#include "../LSH.h"
#include "Node.h"
#include "cnpy.h"
#include <unordered_map>
#include <vector>

namespace hieu {
/////////////////////////////////////////////////////////////
class Layer {
protected:
  std::vector<Node> _nodes;
  std::vector<float> _weights;
  std::vector<float> _bias;
  size_t _layerIdx, _numNodes, _prevNumNodes;

  LSH *_hashTables = nullptr;
  DensifiedWtaHash *_dwtaHasher = nullptr;

public:
  Layer(size_t layerIdx, size_t numNodes, size_t prevNumNodes,
        size_t maxBatchsize, bool sparsify, size_t K, size_t L,
        size_t RangePow, NodeType type);
  virtual ~Layer();

  void Load(const cnpy::npz_t &npzArray);

  virtual size_t computeActivation(std::vector<float> &dataOut,
                                   const std::vector<float> &dataIn) const;

  size_t getNumNodes() const { return _nodes.size(); }
  const Node &getNode(size_t idx) const { return _nodes.at(idx); }
  Node &getNode(size_t idx) { return _nodes.at(idx); }
  const std::vector<Node> &getNodes() const { return _nodes; }
  std::vector<Node> &getNodes() { return _nodes; }

  void HashWeights();
};

} // namespace hieu
