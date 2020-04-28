#include "Node.h"
#include "../Config.h"
#include <cassert>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>
#include <algorithm>

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

void VectorStats(const std::string &str, const SubVectorConst<float> &vec)
{
  float min = 999999999;
  float max = -99999999999;
  float sum = 0;
  float sumSq = 0;
  for (size_t i = 0; i < vec.size(); ++i) {
    float val = vec[i];
    if (val < min) min = val;
    if (val > max) max = val;
    sum += val;
    sumSq += (val * val);
  }
  float mean = sum / (float)vec.size();
  float stddev = sumSq/(float)vec.size() - pow(mean, 2.0f);
  stddev = pow(stddev, 0.5);

  cerr << str
    << " min=" << min
    << " max=" << max
    << " mean=" << mean
    << " stddev=" << stddev
    << " sum=" << sum
    << " sumSq=" << sumSq
    << endl;
}

void Node::HashWeights(LSH &hashTables,
                       const HashBase &hasher) const {
  std::vector<int> hashes = hasher.getHash(_weights);
  std::vector<size_t> hashIndices = hashTables.hashesToIndex(hashes);
  hashTables.Add(hashIndices, _idx, true);
  VectorStats("_weights", _weights);
  Print("_weights", _weights);
  Print("hashes", hashes);
  Print("hashIndices", hashIndices);
  cerr << endl;
  // cerr << "hashes1 " << hashes.size() << " " << hashIndices.size() << endl;
}

} // namespace hieu
