#pragma once
#include "../DensifiedWtaHash.h"
#include "../LSH.h"
#include "../Util.h"
#include <stddef.h>

namespace slide {
namespace hieu {
/////////////////////////////////////////////////////////////
class Node {
protected:
  size_t _idx;
  SubVectorConst<float> _weights;
  float &_nodeBias;
  NodeType _type;

public:
  Node(size_t idx, SubVectorConst<float> &nodeWeights, float &nodeBias,
       size_t batchsize, NodeType type);
  virtual ~Node();

  const SubVectorConst<float> &getWeights() const { return _weights; }

  float computeActivation(const std::vector<float> &dataIn) const;

  void backPropagate(std::vector<Node> &prevNodes,
                     const std::vector<int> &prevActiveNodeIdx,
                     float learningRate, size_t batchIdx);
  void backPropagateFirstLayer(const Vec2d<float> &data, float learningRate,
                               size_t batchIdx);
  void HashWeights(LSH &hashTables, const HashBase &hasher) const;
};
} // namespace hieu
} // namespace slide
