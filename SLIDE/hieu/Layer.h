#pragma once
#include "Node.h"
#include <vector>

namespace hieu {
class Layer {
protected:
  std::vector<Node> _nodes;
  std::vector<float> _weights;
  std::vector<float> _bias;

public:
  Layer(size_t numNodes, size_t prevNumNodes);
  virtual ~Layer();

  void computeActivation(const std::vector<int> &inputIndices1,
                         const std::vector<float> &inputValues1) const;
};

class RELULayer : public Layer {
protected:
public:
  RELULayer(size_t numNodes, size_t prevNumNodes);
  virtual ~RELULayer();
};

class SoftmaxLayer : public Layer {
protected:
public:
  SoftmaxLayer(size_t numNodes, size_t prevNumNodes);
  virtual ~SoftmaxLayer();
};

} // namespace hieu
