#include "Layer.h"
#include "../Util.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include "../DensifiedWtaHash.h"
#include "../srp.h"
#include "../WtaHash.h"

using namespace std;

namespace hieu {
Layer::Layer(size_t layerIdx, size_t numNodes, size_t prevNumNodes,
             size_t maxBatchsize, bool sparsify, size_t K, size_t L,
             size_t RangePow, NodeType type)
    : _layerIdx(layerIdx), _numNodes(numNodes), _prevNumNodes(prevNumNodes) {

  _weights.resize(numNodes * prevNumNodes);
  _bias.resize(numNodes);

  if (sparsify) {
    _hashTables = new LSH(K, L, RangePow);
    
    if (HashFunction == 1) {
      _hasher = new WtaHash(K * L, prevNumNodes);
    }
    else if (HashFunction == 2) {
      _hasher = new DensifiedWtaHash(K * L, prevNumNodes);
    }
    else if (HashFunction == 3) {
      //_binids.resize(previousLayerNumOfNodes);
      //_MinHasher = new DensifiedMinhash(_K * _L, previousLayerNumOfNodes);
      //_MinHasher->getMap(previousLayerNumOfNodes, _binids);
    }
    else if (HashFunction == 4) {
      _hasher = new SparseRandomProjection(prevNumNodes, K * L, Ratio);
    }

  }

  _nodes.reserve(numNodes);
  for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
    SubVectorConst<float> nodeWeights(_weights, nodeIdx * prevNumNodes,
                                      prevNumNodes);
    float &nodeBias = _bias.at(nodeIdx);

    _nodes.emplace_back(
        Node(nodeIdx, nodeWeights, nodeBias, maxBatchsize, type));
  }

  cerr << "Created Layer"
       << " layerIdx=" << _layerIdx << " numNodes=" << _nodes.size()
       << " prevNumNodes=" << _prevNumNodes 
       << " sparsify=" << sparsify
      << endl;
}

void Layer::Load(const cnpy::npz_t &npzArray) {
  cnpy::NpyArray weightArr = npzArray.at("w_layer_" + to_string(_layerIdx));
  //Print("weightArr=", weightArr.shape);
  assert(_weights.size() == weightArr.num_vals);
  memcpy(_weights.data(), weightArr.data<float>(),
         sizeof(float) * weightArr.num_vals);

  cnpy::NpyArray biasArr = npzArray.at("b_layer_" + to_string(_layerIdx));
  //Print("biasArr=", biasArr.shape);
  assert(_bias.size() == biasArr.num_vals);
  memcpy(_bias.data(), biasArr.data<float>(), sizeof(float) * biasArr.num_vals);
}

Layer::~Layer() { 
  delete _hashTables; 
  cerr << "Layer stats " << _layerIdx << " "
    << _totActiveNodes << " "
    << _totComputes << " "
    << (float) _totActiveNodes / (float) _totComputes << " "
    << endl;
}

size_t Layer::computeActivation(std::vector<float> &dataOut,
                                const std::vector<float> &dataIn) const {
  // cerr << "computeActivation layer=" << _layerIdx << endl;
  assert(dataIn.size() == _prevNumNodes);

  if (_hashTables) {
    std::vector<int> hashes = _hasher->getHash(dataIn);
    std::vector<int> hashIndices = _hashTables->hashesToIndex(hashes);
    std::vector<const std::vector<int> *> actives =
        _hashTables->retrieveRaw(hashIndices);

    std::unordered_set<int> activeNodesIdx;
    for (const std::vector<int> *v : actives) {
      // Print("v", *v);
      // cerr << v->size() << " ";
      std::copy(v->begin(), v->end(), std::inserter(activeNodesIdx, activeNodesIdx.end()));
    }
    //cerr << "activeNodesIdx=" << activeNodesIdx.size() << endl;
    _totActiveNodes += activeNodesIdx.size();

    dataOut.resize(_numNodes, 0);
    for (int activeNodeIdx : activeNodesIdx) {
      const Node &node = getNode(activeNodeIdx);
      dataOut.at(activeNodeIdx) = node.computeActivation(dataIn);
    }

  } else {
    dataOut.resize(_numNodes);
    for (size_t nodeIdx = 0; nodeIdx < _nodes.size(); ++nodeIdx) {
      const Node &node = getNode(nodeIdx);
      dataOut.at(nodeIdx) = node.computeActivation(dataIn);
    }
    _totActiveNodes += _nodes.size();
  }

  ++_totComputes;

}

void Layer::HashWeights() {
  if (_hashTables) {
    for (Node &node : _nodes) {
      node.HashWeights(*_hashTables, *_hasher);
    }
  }
}

} // namespace hieu
