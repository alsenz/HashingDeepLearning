#pragma once
#include <vector>
#include "Layer.h"

namespace hieu {
class Network {
protected:
  std::vector<Layer> _layers;

public:
  Network();
  virtual ~Network();
};

} // namespace hieu
