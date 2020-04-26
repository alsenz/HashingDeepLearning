#include "Node.h"
#include "../Config.h"
#include <cassert>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Node::Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias,
           size_t maxBatchsize)
    : _idx(idx), _weights(nodeWeights), _nodeBias(nodeBias) {
  // cerr << "Create Node" << endl;
}

Node::~Node() {}

float Node::computeActivation(const std::vector<float> &dataIn) const {
  assert(dataIn.size() == _weights.size());
  float ret = _nodeBias;
  for (size_t idx = 0; idx < _weights.size(); ++idx) {
    // ret += _nodeWeights[idx] * inVal;
  }

  return ret;
}

void Node::HashWeights(LSH &hashTables,
                       const DensifiedWtaHash &dwtaHasher) const {
  std::vector<int> hashes = dwtaHasher.getHashEasy(_weights);
  std::vector<int> hashIndices = hashTables.hashesToIndex(hashes);
  // Print("hashIndices", hashIndices);
  hashTables.add(hashIndices, _idx);
}

} // namespace hieu
