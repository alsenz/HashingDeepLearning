#include "hieu-main.h"
#include "../Config.h"
#include "../Util.h"
#include "Network.h"
#include "cnpy.h"
#include <fstream>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

namespace hieu {
void EvalDataSVM(int numBatchesTest, Network &mynet, const std::string &path,
                 int epoch, size_t maxBatchsize, size_t inputDim) {
  int totCorrect = 0;
  std::ifstream file(path);
  if (!file) {
    cout << "Error file not found: " << path << endl;
  }

  string str;
  // Skipe header
  std::getline(file, str);

  for (int i = 0; i < numBatchesTest; i++) {
    Vec2d<float> data;
    Vec2d<int> labels;

    CreateData(file, data, labels, maxBatchsize, inputDim);

    int num_features = 0, num_labels = 0;
    for (int batchIdx = 0; batchIdx < maxBatchsize; batchIdx++) {
      num_features += data[batchIdx].size();
      num_labels += labels[batchIdx].size();
    }

    std::cout << maxBatchsize << " records, with " << num_features
              << " features and " << num_labels << " labels" << std::endl;
    size_t correctPredict = mynet.predictClass(data, labels);
    totCorrect += correctPredict;
    std::cout << " iter " << i << ": "
              << totCorrect * 1.0 / (maxBatchsize * (i + 1)) << " correct"
              << std::endl;
  }
  file.close();
  cout << "over all " << totCorrect * 1.0 / (numBatchesTest * maxBatchsize)
       << endl;
}

int main(size_t maxBatchsize, const std::vector<int> &K, const std::vector<int> &L, const std::vector<int> &RangePow) {
  cerr << "Starting" << endl;
  size_t inputDim = 135909;
  size_t numEpochs = 1;
  size_t totRecords = 490449;
  size_t totRecordsTest = 153025;
  int numBatches = totRecords / maxBatchsize;
  int numBatchesTest = totRecordsTest / maxBatchsize;

  hieu::Network mynet(maxBatchsize, K, L, RangePow);
  if (LOADWEIGHT) {
    cnpy::npz_t npzArray = cnpy::npz_load("../savedWeight.npz");
    cerr << "npzArray=" << npzArray.size() << endl;
    mynet.Load(npzArray);
  }

  for (size_t epoch = 0; epoch < numEpochs; epoch++) {
    cerr << "epoch=" << epoch << endl;

    // ReadDataSVM(numBatches, mynet, "../dataset/Amazon/amazon_train.txt",
    // epoch,
    //  maxBatchsize, inputDim);

    EvalDataSVM(20, mynet, "../dataset/Amazon/amazon_test.txt", epoch,
                maxBatchsize, inputDim);
  }

  cerr << "Finished" << endl;
  exit(0);
}
} // namespace hieu
