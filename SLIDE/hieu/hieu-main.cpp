#include "hieu-main.h"
#include "../Util.h"
#include "../Config.h"
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
  }
}

void ReadDataSVM(size_t numBatches, Network &mynet, const std::string &path,
                 int epoch, size_t maxBatchsize, size_t inputDim) {
  std::ifstream file(path);
  if (!file) {
    cout << "Error file not found: " << path << endl;
  }

  std::string str;
  // skipe header
  std::getline(file, str);
  for (size_t i = 0; i < numBatches; i++) {
    Vec2d<float> data;
    Vec2d<int> labels;

    CreateData(file, data, labels, maxBatchsize, inputDim);

    bool rehash = true;
    bool rebuild = true;

    // logloss
    mynet.ProcessInput(data, labels, epoch * numBatches + i, rehash, rebuild);
  }
}

int main(int argc, char *argv[]) {
  cerr << "Starting" << endl;
  size_t inputDim = 135909;
  size_t numEpochs = 5;
  size_t maxBatchsize = 128;
  size_t totRecords = 490449;
  size_t totRecordsTest = 153025;
  int numBatches = totRecords / maxBatchsize;
  int numBatchesTest = totRecordsTest / maxBatchsize;

  cnpy::npz_t npzArray;
  if (LOADWEIGHT) {
    npzArray = cnpy::npz_load("../savedWeight.npz");
    cerr << "npzArray=" << npzArray.size() << endl;
  }

  hieu::Network mynet(maxBatchsize, npzArray);

  for (size_t epoch = 0; epoch < numEpochs; epoch++) {
    cerr << "epoch=" << epoch << endl;

    //ReadDataSVM(numBatches, mynet, "../dataset/Amazon/amazon_train.txt", epoch,
    //  maxBatchsize, inputDim);

    EvalDataSVM(20, mynet, "../dataset/Amazon/amazon_test.txt", epoch,
      maxBatchsize, inputDim);
  }

  cerr << "Finished" << endl;
  exit(0);
}
} // namespace hieu
