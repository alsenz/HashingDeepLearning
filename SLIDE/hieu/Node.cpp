#include "Node.h"
#include "../Config.h"
#include <cassert>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Node::Node(size_t idx, SubVectorConst<float> &nodeWeights, float &nodeBias,
           size_t maxBatchsize, NodeType type)
    : _idx(idx), _weights(nodeWeights), _nodeBias(nodeBias), _type(type) {
  // cerr << "Create Node" << endl;
}

Node::~Node() {}

float Node::computeActivation(const std::vector<float> &dataIn) const {
  assert(dataIn.size() == _weights.size());
  float ret = _nodeBias;
  for (size_t idx = 0; idx < _weights.size(); ++idx) {
    ret += dataIn.at(idx) * _weights.at(idx);
  }
  return ret;
}

void Node::HashWeights(LSH &hashTables,
                       const HashBase &hasher) const {
  std::vector<int> hashes = hasher.getHash(_weights);
  std::vector<int> hashIndices = hashTables.hashesToIndex(hashes);
  hashTables.Add(hashIndices, _idx, true);
  // Print("hashIndices", hashIndices);
  // cerr << "hashes1 " << hashes.size() << " " << hashIndices.size() << endl;
}

} // namespace hieu
