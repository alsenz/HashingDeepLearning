#include "Node.h"
#include <random>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <sys/mman.h>
#include "Config.h"

using namespace std;

void Node::Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *allWeights, float bias, float *allAdamAvgMom, float *allAdamAvgVel, std::vector<train> &train_blob)
{
    _dim = dim;
    _IDinLayer = nodeID;
    _type = type;
    _layerNum = layerID;
    _currentBatchsize = batchsize;

    if (ADAM)
    {
        _adamAvgMom = allAdamAvgMom + dim * nodeID;
        _adamAvgVel = allAdamAvgVel + dim * nodeID;
        _t.resize(_dim);
    }

    _train = SubVector<train>(train_blob.data(), nodeID * batchsize, batchsize);
    _activeInputs = 0;

    _weights = allWeights + dim * nodeID;
    _bias = bias;
    _mirrorbias = _bias;

}

float Node::getLastActivation(int inputID)
{
	if(_train._ptr[inputID]._ActiveinputIds != 1)
		return 0.0;
	return _train._ptr[inputID]._lastActivations;
}


void Node::incrementDelta(int inputID, float incrementValue)
{
	assert(("Input Not Active but still called !! BUG", _train._ptr[inputID]._ActiveinputIds == 1));
	if (_train._ptr[inputID]._lastActivations > 0)
	    _train._ptr[inputID]._lastDeltaforBPs += incrementValue;
}

bool Node::getInputActive(int inputID)
{
    return _train._ptr[inputID]._ActiveinputIds == 1;
}

bool Node::getActiveInputs(void)
{
    return _activeInputs > 0;
}

float Node::getActivation(int* indices, float* values, int length, int inputID)
{
	assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));

	//FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
	if (_train._ptr[inputID]._ActiveinputIds != 1) {
	    _train._ptr[inputID]._ActiveinputIds = 1; //activate input
	    _activeInputs++;
	}

	_train._ptr[inputID]._lastActivations = 0;
	for (int i = 0; i < length; i++)
	{
	    _train._ptr[inputID]._lastActivations += _weights[indices[i]] * values[i];
	}
	_train._ptr[inputID]._lastActivations += _bias;

	switch (_type)
	{
	case NodeType::ReLU:
		if (_train._ptr[inputID]._lastActivations < 0) {
		    _train._ptr[inputID]._lastActivations = 0;
		    _train._ptr[inputID]._lastGradients = 1;
		    _train._ptr[inputID]._lastDeltaforBPs = 0;

        }else{
            _train._ptr[inputID]._lastGradients = 0;
		}
		break;
	case NodeType::Softmax:

		break;
	default:
		cout << "Invalid Node type from Constructor" <<endl;
		break;
	}

	return _train._ptr[inputID]._lastActivations;
}


void Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)
{
	assert(("Input Not Active but still called !! BUG", _train._ptr[inputID]._ActiveinputIds ==1));

	_train._ptr[inputID]._lastActivations /= normalizationConstant + 0.0000001;

	//TODO:check  gradient
	_train._ptr[inputID]._lastGradients = 1;
	if (find (label, label+labelsize, _IDinLayer)!= label+labelsize) {
	    _train._ptr[inputID]._lastDeltaforBPs = (1.0/labelsize - _train._ptr[inputID]._lastActivations) / _currentBatchsize;
	}
	else {
	    _train._ptr[inputID]._lastDeltaforBPs = (-_train._ptr[inputID]._lastActivations) / _currentBatchsize;
	}
}


void Node::backPropagate(std::vector<Node> &previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _train._ptr[inputID]._ActiveinputIds == 1));
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
	    Node &prev_node = previousNodes[previousLayerActiveNodeIds[i]];
	    prev_node.incrementDelta(inputID, _train._ptr[inputID]._lastDeltaforBPs * _weights[previousLayerActiveNodeIds[i]]);

		float grad_t = _train._ptr[inputID]._lastDeltaforBPs * prev_node.getLastActivation(inputID);

		if (ADAM)
		{
			_t[previousLayerActiveNodeIds[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _train._ptr[inputID]._lastDeltaforBPs;
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
    {
        _mirrorbias += learningRate * _train._ptr[inputID]._lastDeltaforBPs;
    }

	_train._ptr[inputID]._ActiveinputIds = 0;
	_train._ptr[inputID]._lastDeltaforBPs = 0;
	_train._ptr[inputID]._lastActivations = 0;
	_activeInputs--;

}


void Node::backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _train._ptr[inputID]._ActiveinputIds == 1));
	for (int i = 0; i < nnzSize; i++)
	{
		float grad_t = _train._ptr[inputID]._lastDeltaforBPs * nnzvalues[i];
		float grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[nnzindices[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[nnzindices[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _train._ptr[inputID]._lastDeltaforBPs;
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
	{
		_mirrorbias += learningRate * _train._ptr[inputID]._lastDeltaforBPs;
	}

	_train._ptr[inputID]._ActiveinputIds = 0;//deactivate inputIDs
	_train._ptr[inputID]._lastDeltaforBPs = 0;
	_train._ptr[inputID]._lastActivations = 0;
    _activeInputs--;
}

void Node::SetlastActivation(int inputID, float realActivation)
{
    _train._ptr[inputID]._lastActivations = realActivation;
}

Node::~Node()
{

	delete[] _indicesInTables;
	delete[] _indicesInBuckets;

	if (ADAM)
	{
		delete[] _adamAvgMom;
		delete[] _adamAvgVel;
	}
}


// for debugging gradients.
float Node::purturbWeight(int weightid, float delta)
{
	_weights[weightid] += delta;
	return _weights[weightid];
}


float Node::getGradient(int weightid, int inputID, float InputVal)
{
	return -_train._ptr[inputID]._lastDeltaforBPs * InputVal;
}
