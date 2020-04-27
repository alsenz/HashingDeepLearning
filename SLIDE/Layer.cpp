#include "Layer.h"
#include "Config.h"
#include <algorithm>
#include <bitset>
#include <climits>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <vector>

using namespace std;

Layer::Layer(size_t noOfNodes, int previousLayerNumOfNodes, int layerID,
             NodeType type, int batchsize, int K, int L, int RangePow,
             float Sparsity)
    : _layerID(layerID), _noOfNodes(noOfNodes), _type(type), _K(K), _L(L),
      _batchsize(batchsize), _RangeRow(RangePow),
      _previousLayerNumOfNodes(previousLayerNumOfNodes),
      _noOfActive(floor(noOfNodes * Sparsity)), _randNode(noOfNodes),
      _hashTables(K, L, RangePow), _Nodes(noOfNodes) {
  // create a list of random nodes just in case not enough nodes from hashtable
  // for active nodes.
  for (size_t n = 0; n < _noOfNodes; n++) {
    _randNode[n] = n;
  }

  std::random_shuffle(_randNode.begin(), _randNode.end());

  // TODO: Initialize Hash Tables and add the nodes. Done by Beidi

  if (HashFunction == 1) {
    _wtaHasher = new WtaHash(_K * _L, previousLayerNumOfNodes);
  } else if (HashFunction == 2) {
    _binids.resize(previousLayerNumOfNodes);
    _dwtaHasher = new DensifiedWtaHash(_K * _L, previousLayerNumOfNodes);
  } else if (HashFunction == 3) {
    _binids.resize(previousLayerNumOfNodes);
    _MinHasher = new DensifiedMinhash(_K * _L, previousLayerNumOfNodes);
    _MinHasher->getMap(previousLayerNumOfNodes, _binids);
  } else if (HashFunction == 4) {
    _srp = new SparseRandomProjection(previousLayerNumOfNodes, _K * _L, Ratio);
  }

  if (LOADWEIGHT) {
    /*
_weights = weights;
_bias = bias;

if (ADAM){
    _adamAvgMom = adamAvgMom;
    _adamAvgVel = adamAvgVel;
}
*/
  } else {
    _weights.resize(_noOfNodes * previousLayerNumOfNodes, 1);
    _bias.resize(_noOfNodes, 1);
    random_device rd;
    default_random_engine dre(rd());
    normal_distribution<float> distribution(0.0, 0.01);

    generate(_weights.begin(), _weights.end(),
             [&]() { return distribution(dre); });
    generate(_bias.begin(), _bias.end(), [&]() { return distribution(dre); });
    // Print(_bias);

    if (ADAM) {
      _adamAvgMom.resize(_noOfNodes * previousLayerNumOfNodes);
      _adamAvgVel.resize(_noOfNodes * previousLayerNumOfNodes);
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  // create nodes for this layer
#pragma omp parallel for // num_threads(1)
  for (size_t i = 0; i < noOfNodes; i++) {
    _Nodes[i].Update(previousLayerNumOfNodes, i, _layerID, type, batchsize,
                     _weights, _bias[i], _adamAvgMom, _adamAvgVel);
    addtoHashTable(_Nodes[i].weights(), i);
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  auto timeDiffInMiliseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  // std::cout << noOfNodes << " " << 1.0 * timeDiffInMiliseconds << std::endl;

  if (type == NodeType::Softmax) {
    _normalizationConstants.resize(batchsize);
  }
}

void Layer::updateTable() {
  if (HashFunction == 1) {
    delete _wtaHasher;
    _wtaHasher = new WtaHash(_K * _L, _previousLayerNumOfNodes);
  } else if (HashFunction == 2) {
    delete _dwtaHasher;
    _binids.resize(_previousLayerNumOfNodes);
    _dwtaHasher = new DensifiedWtaHash(_K * _L, _previousLayerNumOfNodes);
  } else if (HashFunction == 3) {
    delete _MinHasher;
    _binids.resize(_previousLayerNumOfNodes);
    _MinHasher = new DensifiedMinhash(_K * _L, _previousLayerNumOfNodes);
    _MinHasher->getMap(_previousLayerNumOfNodes, _binids);
  } else if (HashFunction == 4) {
    _srp = new SparseRandomProjection(_previousLayerNumOfNodes, _K * _L, Ratio);
  }
}

void Layer::updateRandomNodes() {
  std::random_shuffle(_randNode.begin(), _randNode.end());
}

void Layer::addtoHashTable(SubVector<float> &weights, int ID) {
  // LSH logic
  std::vector<int> hashes;
  if (HashFunction == 1) {
    hashes = _wtaHasher->getHash(weights);
  } else if (HashFunction == 2) {
    hashes = _dwtaHasher->getHashEasy(weights);
  } else if (HashFunction == 3) {
    hashes = _MinHasher->getHashEasy(_binids, weights, TOPK);
  } else if (HashFunction == 4) {
    hashes = _srp->getHash(weights);
  }

  std::vector<int> hashIndices = _hashTables.hashesToIndex(hashes);
  _hashTables.add(hashIndices, ID + 1, false);

  // mine
  /*
  std::vector<float> w(135909, 0);
  hashes = _dwtaHasher->getHashEasy(w);
  hashIndices = _hashTables.hashesToIndex(hashes);

  //cerr << "w" << endl; Print(w);
  cerr << "hashes" << endl; Print(hashes);
  cerr << "hashIndices" << endl; Print(hashIndices);

  random_device rd;
  default_random_engine dre(rd());
  normal_distribution<float> distribution(0.0, 0.01);
  generate(w.begin(), w.end(),[&]() { return distribution(dre); });
  //w.clear();
  //w.resize(135909, 44);
  hashes = _dwtaHasher->getHashEasy(w);
  hashIndices = _hashTables.hashesToIndex(hashes);

  //cerr << "w" << endl; Print(w);
  cerr << "hashes" << endl; Print(hashes);
  cerr << "hashIndices" << endl; Print(hashIndices);

  exit(44);
  */
}

Node &Layer::getNodebyID(size_t nodeID) {
  assert(("nodeID less than _noOfNodes", nodeID < _noOfNodes));
  return _Nodes[nodeID];
}

std::vector<Node> &Layer::getAllNodes() { return _Nodes; }

float Layer::getNomalizationConstant(int inputID) const {
  assert(("Error Call to Normalization Constant for non - softmax layer",
          _type == NodeType::Softmax));
  return _normalizationConstants[inputID];
}

float innerproduct(const int *index1, const float *value1, int len1,
                   const float *value2) {
  float total = 0;
  for (int i = 0; i < len1; i++) {
    total += value1[i] * value2[index1[i]];
  }
  return total;
}

float collision(int *hashes, int *table_hashes, int k, int l) {
  int cp = 0;
  for (int i = 0; i < l; i = i + k) {
    int tmp = 0;
    for (int j = i; j < i + k; j++) {
      if (hashes[j] == table_hashes[j]) {
        tmp++;
      }
    }
    if (tmp == k) {
      cp++;
    }
  }
  return cp * 1.0 / (l / k);
}

int Layer::queryActiveNodeandComputeActivations(
    Vec2d<int> &activenodesperlayer, Vec2d<float> &activeValuesperlayer,
    int inputID, const std::vector<int> &label, float Sparsity, int iter,
    bool train) {
  // LSH QueryLogic
  // Beidi. Query out all the candidate nodes
  int len;
  int in = 0;

  if (Sparsity == 1.0) {
    len = _noOfNodes;
    activenodesperlayer[_layerID + 1].resize(len); // assuming not intitialized;
    for (int i = 0; i < len; i++) {
      activenodesperlayer[_layerID + 1][i] = i;
    }
  } else {
    if (Mode == 1) {
      std::vector<int> hashes;
      if (HashFunction == 1) {
        hashes = _wtaHasher->getHash(activeValuesperlayer[_layerID]);
      } else if (HashFunction == 2) {
        hashes = _dwtaHasher->getHash(activenodesperlayer[_layerID],
                                      activeValuesperlayer[_layerID]);
      } else if (HashFunction == 3) {
        hashes = _MinHasher->getHashEasy(_binids,
                                         activeValuesperlayer[_layerID], TOPK);
      } else if (HashFunction == 4) {
        hashes = _srp->getHashSparse(activenodesperlayer[_layerID],
                                     activeValuesperlayer[_layerID]);
      }
      std::vector<int> hashIndices = _hashTables.hashesToIndex(hashes);
      std::vector<const std::vector<int> *> actives =
          _hashTables.retrieveRaw(hashIndices);

      // Get candidates from hashtable
      auto t00 = std::chrono::high_resolution_clock::now();

      std::map<int, size_t> counts;
      // Make sure that the true label node is in candidates
      if (_type == NodeType::Softmax) {
        if (train) {
          for (int i = 0; i < label.size(); i++) {
            counts[label[i]] = _L;
          }
        }
      }

      for (int i = 0; i < _L; i++) {
        assert(actives[i]);
        for (int j = 0; j < actives[i]->size(); j++) {
          int tempID = (*actives[i])[j] - 1;
          assert(tempID >= 0);
          counts[tempID] += 1;
        }
      }
      auto t11 = std::chrono::high_resolution_clock::now();

      // thresholding
      auto t3 = std::chrono::high_resolution_clock::now();
      vector<int> vect;
      for (auto &&x : counts) {
        if (x.second > THRESH) {
          vect.push_back(x.first);
        }
      }

      len = vect.size();
      activenodesperlayer[_layerID + 1].resize(len);

      for (int i = 0; i < len; i++) {
        activenodesperlayer[_layerID + 1][i] = vect[i];
      }
      auto t33 = std::chrono::high_resolution_clock::now();
      in = len;
    }
    if (Mode == 4) {
      // Print("activenodesperlayer", activenodesperlayer[_layerID]);
      // Print("activeValuesperlayer", activeValuesperlayer[_layerID]);
      std::vector<int> hashes;
      if (HashFunction == 1) {
        hashes = _wtaHasher->getHash(activeValuesperlayer[_layerID]);
      } else if (HashFunction == 2) {
        hashes = _dwtaHasher->getHash(activenodesperlayer[_layerID],
                                      activeValuesperlayer[_layerID]);
      } else if (HashFunction == 3) {
        hashes = _MinHasher->getHashEasy(_binids,
                                         activeValuesperlayer[_layerID], TOPK);
      } else if (HashFunction == 4) {
        hashes = _srp->getHashSparse(activenodesperlayer[_layerID],
                                     activeValuesperlayer[_layerID]);
      }
      std::vector<int> hashIndices = _hashTables.hashesToIndex(hashes);
      std::vector<const std::vector<int> *> actives =
          _hashTables.retrieveRaw(hashIndices);
      // we now have a sparse array of indices of active nodes

      // Get candidates from hashtable
      std::map<int, size_t> counts;
      // Make sure that the true label node is in candidates
      if (_type == NodeType::Softmax && train) {
        for (int i = 0; i < label.size(); i++) {
          counts[label[i]] = _L;
        }
      }

      for (int i = 0; i < _L; i++) {
        assert(actives[i]);
        // copy sparse array into (dense) map
        for (int j = 0; j < actives[i]->size(); j++) {
          int tempID = (*actives[i])[j] - 1;
          assert(tempID >= 0);
          counts[tempID] += 1;
        }
      }

      in = counts.size();
      if (counts.size() < 1500) {
        size_t start = rand() % _noOfNodes;
        for (size_t i = start; i < _noOfNodes; i++) {
          if (counts.size() >= 1000) {
            break;
          }
          if (counts.count(_randNode[i]) == 0) {
            counts[_randNode[i]] = 0;
          }
        }

        if (counts.size() < 1000) {
          for (size_t i = 0; i < _noOfNodes; i++) {
            if (counts.size() >= 1000) {
              break;
            }
            if (counts.count(_randNode[i]) == 0) {
              counts[_randNode[i]] = 0;
            }
          }
        }
      }

      len = counts.size();
      activenodesperlayer[_layerID + 1].resize(len);

      // copy map into new array
      int i = 0;
      for (auto &&x : counts) {
        activenodesperlayer[_layerID + 1][i] = x.first;
        i++;
      }
    } else if (Mode == 2 & _type == NodeType::Softmax) {
      len = floor(_noOfNodes * Sparsity);
      activenodesperlayer[_layerID + 1].resize(len);

      auto t1 = std::chrono::high_resolution_clock::now();
      bitset<MAPLEN> bs;
      int tmpsize = 0;
      if (_type == NodeType::Softmax) {
        if (train) {
          for (int i = 0; i < label.size(); i++) {
            activenodesperlayer[_layerID + 1][i] = label[i];
            bs[label[i]] = 1;
          }
          tmpsize = label.size();
        }
      }

      while (tmpsize < len) {
        int v = rand() % _noOfNodes;
        if (bs[v] == 0) {
          activenodesperlayer[_layerID + 1][tmpsize] = v;
          bs[v] = 1;
          tmpsize++;
        }
      }

      auto t2 = std::chrono::high_resolution_clock::now();
      //            std::cout << "sampling "<<" takes" << 1.0 *
      //            timeDiffInMiliseconds << std::endl;

    }

    else if (Mode == 3 & _type == NodeType::Softmax) {
      len = floor(_noOfNodes * Sparsity);
      activenodesperlayer[_layerID + 1].resize(len);
      vector<pair<float, int>> sortW;
      int what = 0;

      for (size_t s = 0; s < _noOfNodes; s++) {
        float tmp = innerproduct(activenodesperlayer[_layerID].data(),
                                 activeValuesperlayer[_layerID].data(),
                                 activenodesperlayer[_layerID].size(),
                                 _Nodes[s].weights().data());
        tmp += _Nodes[s].bias();
        if (find(label.begin(), label.end(), s) != label.end()) {
          sortW.push_back(make_pair(-1000000000, s));
          what++;
        } else {
          sortW.push_back(make_pair(-tmp, s));
        }
      }

      std::sort(begin(sortW), end(sortW));

      for (int i = 0; i < len; i++) {
        activenodesperlayer[_layerID + 1][i] = sortW[i].second;
        if (find(label.begin(), label.end(), sortW[i].second) != label.end()) {
          in = 1;
        }
      }
    }
  }

  //***********************************
  // Calc logit and Z for softmax
  activeValuesperlayer[_layerID + 1].resize(
      len); // assuming its not initialized else memory leak;
  float maxValue = 0;
  if (_type == NodeType::Softmax)
    _normalizationConstants[inputID] = 0;

  // find activation for all ACTIVE nodes in layer
  for (int i = 0; i < len; i++) {
    activeValuesperlayer[_layerID + 1][i] =
        _Nodes[activenodesperlayer[_layerID + 1][i]].getActivation(
            activenodesperlayer[_layerID], activeValuesperlayer[_layerID],
            activenodesperlayer[_layerID].size(), inputID);
    if (_type == NodeType::Softmax &&
        activeValuesperlayer[_layerID + 1][i] > maxValue) {
      maxValue = activeValuesperlayer[_layerID + 1][i];
    }
  }

  if (_type == NodeType::Softmax) {
    for (int i = 0; i < len; i++) {
      float realActivation =
          exp(activeValuesperlayer[_layerID + 1][i] - maxValue);
      activeValuesperlayer[_layerID + 1][i] = realActivation;
      _Nodes[activenodesperlayer[_layerID + 1][i]].SetlastActivation(
          inputID, realActivation);
      _normalizationConstants[inputID] += realActivation;
    }
  }

  return in;
}

void Layer::saveWeights(const string &file) {
  if (_layerID == 0) {
    cnpy::npz_save(file, "w_layer_0", _weights.data(),
                   {_noOfNodes, _Nodes[0].dim()}, "w");
    cnpy::npz_save(file, "b_layer_0", _bias.data(), {_noOfNodes}, "a");
    cnpy::npz_save(file, "am_layer_0", _adamAvgMom.data(),
                   {_noOfNodes, _Nodes[0].dim()}, "a");
    cnpy::npz_save(file, "av_layer_0", _adamAvgVel.data(),
                   {_noOfNodes, _Nodes[0].dim()}, "a");
    cout << "save for layer 0" << endl;
    // cout << _weights[0] << " " << _weights[1] << endl;
  } else {
    cnpy::npz_save(file, "w_layer_" + to_string(_layerID), _weights.data(),
                   {_noOfNodes, _Nodes[0].dim()}, "a");
    cnpy::npz_save(file, "b_layer_" + to_string(_layerID), _bias.data(),
                   {_noOfNodes}, "a");
    cnpy::npz_save(file, "am_layer_" + to_string(_layerID), _adamAvgMom.data(),
                   {_noOfNodes, _Nodes[0].dim()}, "a");
    cnpy::npz_save(file, "av_layer_" + to_string(_layerID), _adamAvgVel.data(),
                   {_noOfNodes, _Nodes[0].dim()}, "a");
    cout << "save for layer " << to_string(_layerID) << endl;
    // cout << _weights[0] << " " << _weights[1] << endl;
  }
}

Layer::~Layer() {
  delete _wtaHasher;
  delete _dwtaHasher;
  delete _srp;
  delete _MinHasher;
}
