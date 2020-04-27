#include "Network.h"
#include "../DensifiedWtaHash.h"
#include <iostream>
#include <algorithm>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Network::Network(size_t maxBatchsize) {
  size_t inputDim = 135909;

  cerr << "Create Network" << endl;
  _layers.push_back(new Layer(0, 128, inputDim, maxBatchsize, false, 2, 20, 6));
  _layers.push_back(new Layer(1, 670091, 128, maxBatchsize, true, 6, 50, 18));
}

void Network::Load(const cnpy::npz_t &npzArray) {
  cerr << "Load Network" << endl;
  for (Layer *layer : _layers) {
    layer->Load(npzArray);
  }

  HashWeights();
}

Network::~Network() {
  for (Layer *layer : _layers) {
    delete layer;
  }
}

size_t Network::predictClass(const Vec2d<float> &data,
                             const Vec2d<int> &labels) const {
  assert(data.size() == labels.size());
  size_t batchSize = data.size();

  size_t correctPred = 0;

  // inference
  for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
    const std::vector<float> &data1 = data.at(batchIdx);
    const std::vector<int> &labels1 = labels.at(batchIdx);

    const std::vector<float> *lastActivations =
        computeActivation(data1, labels1);

    size_t correctPred1 = computeCorrect(*lastActivations, labels1);
    correctPred += correctPred1;

    delete lastActivations;
  }
  return correctPred;
}

size_t Network::computeCorrect(const std::vector<float> &lastActivations, const std::vector<int> &labels1) const {  
  float max_act = -222222222;
  int predict_class = -1;
  for (size_t idx = 0; idx < lastActivations.size(); ++idx) {
    float cur_act = lastActivations[idx];
    if (max_act < cur_act) {
      max_act = cur_act;
      predict_class = idx;
    }
  }

  size_t ret = 0;
  if (std::find(labels1.begin(), labels1.end(), predict_class) !=
    labels1.end()) {
    ret++;
  }
  return ret;
}

const std::vector<float> *
Network::computeActivation(const std::vector<float> &data1,
                           const std::vector<int> &labels1) const {
  size_t correctPred = 0;

  std::vector<float> *dataOut = new std::vector<float>;

  const Layer &firstLayer = getLayer(0);
  firstLayer.computeActivation(*dataOut, data1);

  std::vector<float> *dataIn = dataOut;
  dataOut = new std::vector<float>;

  for (int layerIdx = 1; layerIdx < _layers.size(); ++layerIdx) {
    const Layer &layer = getLayer(layerIdx);
    layer.computeActivation(*dataOut, *dataIn);

    std::swap(dataIn, dataOut);
  }
  delete dataOut;

  return dataIn;
}

void Network::HashWeights() {
  cerr << "Start HashWeights()" << endl;
  for (Layer *layer : _layers) {
    layer->HashWeights();
  }
  cerr << "Finished HashWeights()" << endl;
}

} // namespace hieu