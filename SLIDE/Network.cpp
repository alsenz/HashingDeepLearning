#include "Network.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "Config.h"
#include <omp.h>
#define DEBUG 1
using namespace std;


Network::Network(const std::vector<int> &sizesOfLayers, const std::vector<NodeType> &layersTypes, int noOfLayers, int batchSize, float lr, int inputdim, const std::vector<int> &K, const std::vector<int> &L, const std::vector<int> &RangePow, const std::vector<float> &Sparsity, cnpy::npz_t arr)
:_sizesOfLayers(sizesOfLayers)
,_Sparsity(Sparsity)
,_layersTypes(layersTypes)
{
    
    _numberOfLayers = noOfLayers;
    _learningRate = lr;
    _currentBatchSize = batchSize;

    for (int i = 0; i < noOfLayers; i++) {
        int previousLayerNumOfNodes;
        if (i != 0) {
          previousLayerNumOfNodes = sizesOfLayers[i - 1];
        } else {
          previousLayerNumOfNodes = inputdim;
        }
        _hiddenlayers.emplace_back(new Layer(sizesOfLayers[i], previousLayerNumOfNodes, i, _layersTypes[i], _currentBatchSize, K[i], L[i], RangePow[i], Sparsity[i]));
    }
    cout << "after layer" << endl;
}


Layer &Network::getLayer(int LayerID) {
  assert(LayerID < _numberOfLayers);
  return *_hiddenlayers[LayerID];
}

const Layer &Network::getLayer(int LayerID) const {
  assert(LayerID < _numberOfLayers);
  return *_hiddenlayers[LayerID];
}

int Network::predictClass(const std::vector< std::vector<int> > &inputIndices, const vector< vector<float> > &inputValues, const vector<int> &length, const vector< vector<int> > &labels, const vector<int> &labelsize) {
    int correctPred = 0;
    //cerr << "start Network::predictClass " << _currentBatchSize << endl;
    //cerr << "_currentBatchSize=" << _currentBatchSize << endl;
    //cerr << "_numberOfLayers=" << _numberOfLayers << endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < _currentBatchSize; i++) {
      std::vector< std::vector<int> > activenodesperlayer(_numberOfLayers + 1);
      std::vector< vector<float> > activeValuesperlayer(_numberOfLayers + 1);
        std::vector<int> sizes(_numberOfLayers + 1);

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = length[i];

        //inference
        for (int j = 0; j < _numberOfLayers; j++) {
          getLayer(j).queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], 0,
                    _Sparsity[_numberOfLayers+j]);
        }

        //compute softmax
        int noOfClasses = sizes[_numberOfLayers];
        float max_act = -222222222;
        int predict_class = -1;
        for (int k = 0; k < noOfClasses; k++) {
            size_t nodeId = activenodesperlayer[_numberOfLayers][k];
            const Node &node = getLayer(_numberOfLayers - 1).getNodebyID(nodeId);
            float cur_act = node.getLastActivation(i);
            if (max_act < cur_act) {
                max_act = cur_act;
                predict_class = activenodesperlayer[_numberOfLayers][k];
            }
        }

        //cerr << "_numberOfLayers=" << _numberOfLayers << endl;
        //cerr << "noOfClasses=" << noOfClasses << endl;
        //cerr << "predict_class=" << predict_class << endl;
        /*
        cerr << "labels=";
        for (int tt = 0; tt < labelsize[i]; tt++) {
          cerr << labels[i][tt] << " ";
        }
        cerr << endl;
        */
        if (std::find (labels[i].begin(), labels[i].end(), predict_class)!= labels[i].end()) {
            correctPred++;
            //cerr << "correct" << endl;
        }        
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //std::cout << "Inference takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;

    //cerr << "finished Network::predictClass, correctPred=" << correctPred << endl;
    return correctPred;
}


int Network::ProcessInput(const std::vector< std::vector<int> > &inputIndices, const vector< vector<float> > &inputValues, const vector<int> &lengths, const vector< vector<int> > &labels, const vector<int> &labelsize, int iter, bool rehash, bool rebuild) {
    //cerr << "start Network::ProcessInput" << endl;
    float logloss = 0.0;
    int* avg_retrieval = new int[_numberOfLayers]();

    for (int j = 0; j < _numberOfLayers; j++)
        avg_retrieval[j] = 0;


    if(iter%6946==6945 ){
        //_learningRate *= 0.5;
      getLayer(1).updateRandomNodes();
    }
    float tmplr = _learningRate;
    if (ADAM) {
        tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
                (1 - pow(BETA1, iter + 1));
    }else{
//        tmplr *= pow(0.9, iter/10.0);
    }

    vector < vector< vector<int> > > activeNodesPerBatch(_currentBatchSize);      // batch, layer, node
    vector < vector< vector<float> > > activeValuesPerBatch(_currentBatchSize); // batch, layer, node ???
    std::vector < std::vector<int> > sizesPerBatch(_currentBatchSize);
#pragma omp parallel for
    for (int i = 0; i < _currentBatchSize; i++) {
        vector< vector<int> > activenodesperlayer(_numberOfLayers + 1);     // layer, node
        vector< vector<float> > activeValuesperlayer(_numberOfLayers + 1);  // layer, node ???
        std::vector<int> sizes(_numberOfLayers + 1);

        activeNodesPerBatch[i] = activenodesperlayer;
        activeValuesPerBatch[i] = activeValuesperlayer;
        sizesPerBatch[i] = sizes;

        activenodesperlayer[0] = inputIndices[i];  // inputs parsed from training data file
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = lengths[i];
        int in;
        //auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < _numberOfLayers; j++) {
            in = getLayer(j).queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], labelsize[i],
                    _Sparsity[j]);
            avg_retrieval[j] += in;
        }
        /*
        for (int j = 0; j < _numberOfLayers; j++) {
          cerr << "avg_retrieval[" << j << "]=" << avg_retrieval[j] << endl;
          cerr << "_Sparsity=" << _Sparsity[j] << endl;
        }
        */
        //Now backpropagate.
        // layers
        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            Layer &layer = getLayer(j);
            Layer &prev_layer = getLayer(j - 1);
            // nodes
            for (int k = 0; k < sizesPerBatch[i][j + 1]; k++) {
                Node &node = layer.getNodebyID(activeNodesPerBatch[i][j + 1][k]);
                if (j == _numberOfLayers - 1) {
                    //TODO: Compute Extra stats: labels[i];
                    node.ComputeExtaStatsForSoftMax(layer.getNomalizationConstant(i), i, labels[i], labelsize[i]);
                }
                if (j != 0) {
                    node.backPropagate(prev_layer.getAllNodes(), activeNodesPerBatch[i][j], sizesPerBatch[i][j], tmplr, i);
                } else {
                    node.backPropagateFirstLayer(inputIndices[i], inputValues[i], lengths[i], tmplr, i);
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    bool tmpRehash;
    bool tmpRebuild;

    for (int l=0; l<_numberOfLayers ;l++) {
        if(rehash & _Sparsity[l]<1){
            tmpRehash=true;
        }else{
            tmpRehash=false;
        }
        if(rebuild & _Sparsity[l]<1){
            tmpRebuild=true;
        }else{
            tmpRebuild=false;
        }
        if (tmpRehash) {
          getLayer(l).getHashTables().clear();
        }
        if (tmpRebuild){
          getLayer(l).updateTable();
        }
        int ratio = 1;
#pragma omp parallel for
        for (size_t m = 0; m < getLayer(l).getNoOfNodes(); m++)
        {
            Node &tmp = getLayer(l).getNodebyID(m);
            int dim = tmp.getDim();
            vector<float> local_weights(dim);
            std::copy(tmp.getWeights().data(), tmp.getWeights().data() + dim, local_weights.data());

            if(ADAM){
                for (int d=0; d < dim;d++){
                    float _t = tmp.getT(d);
                    float Mom = tmp.getAdamAvgMom(d);
                    float Vel = tmp.getAdamAvgVel(d);
                    Mom = BETA1 * Mom + (1 - BETA1) * _t;
                    Vel = BETA2 * Vel + (1 - BETA2) * _t * _t;
                    local_weights[d] += ratio * tmplr * Mom / (sqrt(Vel) + EPS);
                    tmp.setAdamAvgMom(d, Mom);
                    tmp.setAdamAvgVel(d, Vel);
                    tmp.setT(d, 0);
                }

                tmp.getAdamAvgMombias() = BETA1 * tmp.getAdamAvgMombias() + (1 - BETA1) * tmp.getTBias();
                tmp.getAdamAvgVelbias() = BETA2 * tmp.getAdamAvgVelbias() + (1 - BETA2) * tmp.getTBias() * tmp.getTBias();
                tmp.getBias() += ratio*tmplr * tmp.getAdamAvgMombias() / (sqrt(tmp.getAdamAvgVelbias()) + EPS);
                tmp.getTBias() = 0;
            }
            else
            {
                std::copy(tmp.getMirrorWeights(), tmp.getMirrorWeights()+(tmp.getDim()) , tmp.getWeights().data());
                tmp.getBias() = tmp.getMirrorBias();
            }
            if (tmpRehash) {
                const int *hashes;
                if(HashFunction==1) {
                    hashes = getLayer(l).getWTAHasher().getHash(local_weights);
                }else if (HashFunction==2){
                    hashes = getLayer(l).getDensifiedWtaHash().getHashEasy(local_weights, dim, TOPK);
                }else if (HashFunction==3){
                    hashes = getLayer(l).getDensifiedMinhash().getHashEasy(getLayer(l).getBinIds(), local_weights, dim, TOPK);
                }else if (HashFunction==4){
                    hashes = getLayer(l).getSparseRandomProjection().getHash(local_weights, dim);
                }

                std::vector<int> hashIndices = getLayer(l).getHashTables().hashesToIndex(hashes);
                std::vector<int> bucketIndices = getLayer(l).getHashTables().add(hashIndices, m+1);

                delete[] hashes;
            }

            std::copy(local_weights.begin(), local_weights.end(), tmp.getWeights().data());
        }
    }
    
    if (DEBUG&rehash) {
        cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;
    }

    //cerr << "finished Network::ProcessInput logloss=" << logloss << endl;
    return logloss;
}


void Network::saveWeights(string file)
{
    for (int i=0; i< _numberOfLayers; i++){
      getLayer(i).saveWeights(file);
    }
}


Network::~Network() {
  for (int i = 0; i < _numberOfLayers; i++) {
    delete _hiddenlayers[i];
  }
}
