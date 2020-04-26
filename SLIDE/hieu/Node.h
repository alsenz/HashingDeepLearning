#pragma once
#include "../Util.h"
#include "../DensifiedWtaHash.h"
#include "../LSH.h"
#include <stddef.h>

namespace hieu {
/////////////////////////////////////////////////////////////
class Node {
protected:
  size_t _idx;
  SubVector<float> _weights;
  float &_nodeBias;

public:
  Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias, size_t batchsize);
  virtual ~Node();

  const SubVector<float> &getWeights() const { return _weights; }

  float computeActivation(const std::vector<float> &dataIn) const;

  void backPropagate(std::vector<Node> &prevNodes,
                     const std::vector<int> &prevActiveNodeIdx, float learningRate,
                     size_t batchIdx);
  void backPropagateFirstLayer(const Vec2d<float> &data, float learningRate,
                               size_t batchIdx);
  void HashWeights(LSH &hashTables, const DensifiedWtaHash &dwtaHasher) const;

};
} // namespace hieu
