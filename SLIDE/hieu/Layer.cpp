#include "Layer.h"
#include "../Util.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

namespace hieu {
Layer::Layer(size_t layerIdx, size_t numNodes, size_t prevNumNodes, size_t maxBatchsize, size_t K, size_t L, size_t RangePow)
    : _layerIdx(layerIdx), _numNodes(numNodes), _prevNumNodes(prevNumNodes), _hashTables(K, L, RangePow), _dwtaHasher(K * L, prevNumNodes) {

  _weights.resize(numNodes * prevNumNodes);
  _bias.resize(numNodes);

  /*
  random_device rd;
  default_random_engine dre(rd());
  normal_distribution<float> distribution(0.0, 0.01);

  generate(_weights.begin(), _weights.end(),
           [&]() { return distribution(dre); });
  generate(_bias.begin(), _bias.end(), [&]() { return distribution(dre); });
  */

  _nodes.reserve(numNodes);
  for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
    SubVector<float> nodeWeights =
        SubVector<float>(_weights, nodeIdx * prevNumNodes, prevNumNodes);
    float &nodeBias = _bias.at(nodeIdx);

    _nodes.emplace_back(Node(nodeIdx, nodeWeights, nodeBias, maxBatchsize));
  }

  cerr << "Created Layer"
       << " layerIdx=" << _layerIdx << " numNodes=" << _nodes.size()
       << " prevNumNodes=" << _prevNumNodes << endl;
}

Layer::Layer(size_t layerIdx, size_t numNodes, size_t prevNumNodes, size_t maxBatchsize, size_t K, size_t L, size_t RangePow, const cnpy::npz_t &npzArray)
  : Layer(layerIdx, numNodes, prevNumNodes, maxBatchsize, K, L, RangePow)
{
  cnpy::NpyArray weightArr, biasArr;

  weightArr = npzArray.at("w_layer_" + to_string(layerIdx));
  Print("weightArr=", weightArr.shape);
  assert(_weights.size() == weightArr.num_vals);
  //memcpy(_weights.data(), weightArr.data<float>(), sizeof(float) * weightArr.num_vals);

  biasArr = npzArray.at("b_layer_" + to_string(layerIdx));
  Print("biasArr=", biasArr.shape);
  assert(_bias.size() == biasArr.num_vals);
  //memcpy(_bias.data(), biasArr.data<float>(), sizeof(float) * biasArr.num_vals);
}

Layer::~Layer() {}

size_t Layer::computeActivation(std::vector<float> &dataOut,
                                const std::vector<float> &dataIn) const {
  assert(dataIn.size() == _prevNumNodes);
  dataOut.resize(_numNodes);
  for (size_t nodeIdx = 0; nodeIdx < _nodes.size(); ++nodeIdx) {
    const Node &node = getNode(nodeIdx);
    dataOut.at(nodeIdx) = node.computeActivation(dataIn);
  }
}

void Layer::HashWeights()
{
  for (Node &node : _nodes) {

  }
}

} // namespace hieu