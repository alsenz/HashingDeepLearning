#pragma once
#include "Layer.h"
#include "cnpy.h"
#include <unordered_map>
#include <vector>

namespace slide {
namespace hieu {
class Network {
protected:
  std::vector<Layer *> _layers;

  const Layer &getLayer(size_t idx) const { return *_layers.at(idx); }
  Layer &getLayer(size_t idx) { return *_layers.at(idx); }

  const std::vector<float> *
  computeActivation(const std::vector<float> &data1,
                    const std::vector<int> &labels) const;
  size_t computeCorrect(const std::vector<float> &lastActivations,
                        const std::vector<int> &labels1) const;

public:
  Network(size_t maxBatchsize, const std::vector<int> &K,
          const std::vector<int> &L, const std::vector<int> &RangePow,
          const std::vector<float> &Sparsity);
  virtual ~Network();

  void Load(const cnpy::npz_t &npzArray);

  size_t predictClass(const Vec2d<float> &data, const Vec2d<int> &labels) const;

  void HashWeights();
};

} // namespace hieu
} // namespace slide
