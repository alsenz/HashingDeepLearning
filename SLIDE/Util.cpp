#include "Util.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

using namespace std;

void CreateData(std::ifstream &file, Vec2d<float> &data, Vec2d<int> &labels,
                int maxBatchsize, size_t inputDim) {
  data.resize(maxBatchsize);
  labels.resize(maxBatchsize);

  Vec2d<int> records(maxBatchsize);
  Vec2d<float> values(maxBatchsize);

  CreateData(file, records, values, labels, maxBatchsize);
  assert(records.size() == values.size());
  assert(records.size() == labels.size());

  for (size_t batchIdx = 0; batchIdx < records.size(); ++batchIdx) {
    const std::vector<int> &records1 = records[batchIdx];
    const std::vector<float> &values1 = values[batchIdx];
    assert(records1.size() == values1.size());
    //Print("records1", records1);
    //Print("values1", values1);

    std::vector<float> &data1 = data[batchIdx];
    data1.resize(inputDim, 0);
    for (size_t featureIdx = 0; featureIdx < records1.size(); ++featureIdx) {
      int feature = records1[featureIdx];
      float value = values1[featureIdx];
      data1.at(feature) = value;
    }
  }
}

void CreateData(std::ifstream &file, Vec2d<int> &records, Vec2d<float> &values,
                Vec2d<int> &labels, int maxBatchsize) {
  int nonzeros = 0;
  int count = 0;
  vector<string> list;
  vector<string> value;
  vector<string> label;
  string str;
  while (std::getline(file, str)) {
    char *mystring = &str[0];
    char *pch, *pchlabel;
    int track = 0;
    list.clear();
    value.clear();
    label.clear();
    pch = strtok(mystring, " ");
    pch = strtok(NULL, " :");
    while (pch != NULL) {
      if (track % 2 == 0)
        list.push_back(pch);
      else if (track % 2 == 1)
        value.push_back(pch);
      track++;
      pch = strtok(NULL, " :");
    }

    pchlabel = strtok(mystring, ",");
    while (pchlabel != NULL) {
      label.push_back(pchlabel);
      pchlabel = strtok(NULL, ",");
    }

    nonzeros += list.size();
    records[count].resize(list.size());
    values[count].resize(list.size());
    labels[count] = std::vector<int>(label.size());

    int currcount = 0;
    vector<string>::iterator it;
    for (it = list.begin(); it < list.end(); it++) {
      records[count][currcount] = stoi(*it);
      currcount++;
    }
    currcount = 0;
    for (it = value.begin(); it < value.end(); it++) {
      values[count][currcount] = stof(*it);
      currcount++;
    }
    currcount = 0;
    for (it = label.begin(); it < label.end(); it++) {
      labels[count][currcount] = stoi(*it);
      currcount++;
    }

    count++;
    if (count >= maxBatchsize)
      break;
  }
}
