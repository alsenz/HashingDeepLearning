#pragma once
#include "Util.h"
#include <vector>

class HashBase {
public:
  virtual ~HashBase() {}
  virtual std::vector<int> getHash(const std::vector<float> &data) const = 0;
  virtual std::vector<int> getHash(const SubVectorConst<float> &data) const = 0;
};