#include "Node.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "Network.h"
#include <algorithm>
#include <map>
#include <climits>
#include <cstring>
#include <cfloat>
#include<complex>
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>
#include "Config.h"
#include "Util.h"

int Batchsize = 1000;
int Rehash = 1000;
int Rebuild = 1000;
int InputDim = 784;
int totRecords = 60000;
int totRecordsTest = 10000;
float Lr = 0.0001;
int Epoch = 5;
int Stepsize = 20;
int numLayer = 3;
string trainData = "";
string testData = "";
string Weights = "";
string savedWeights = "";
string logFile = "";
int globalTime = 0;


std::vector<int> sizesOfLayers;
std::vector<int> RangePow;
std::vector<int> K;
std::vector<int> L;
std::vector<float> Sparsity;

using namespace std;

#define ALL(c) c.begin(), c.end()
#define FOR(i,c) for(typeof(c.begin())i=c.begin();i!=c.end();++i)
#define REP(i,n) for(int i=0;i<n;++i)
#define fst first
#define snd second


void endianSwap(unsigned int &x) {
    x = (x>>24)|((x<<8)&0x00FF0000)|((x>>8)&0x0000FF00)|(x<<24);
}
typedef vector<unsigned int> Image;


string trim(string& str)
{
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}


void parseconfig(string filename)
{
    std::ifstream file(filename);
    if(!file)
    {
        cout<<"Error Config file not found: Given Filename "<< filename << endl;
    }
    std::string str;
    while (getline(file, str))
    {
        if (str == "")
            continue;

        std::size_t found = str.find("#");
        if (found != std::string::npos)
            continue;

        if (trim(str).length() < 3)
            continue;

        int index = str.find_first_of("=");
        string first = str.substr(0, index);
        string second = str.substr(index + 1, str.length());

        if (trim(first) == "RangePow")
        {
            string str = trim(second).c_str();
            RangePow.resize(numLayer);
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                RangePow[i] = atoi(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "K")
        {
          string str = trim(second).c_str();
            K.resize(numLayer);
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                K[i] = atoi(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "L")
        {
          string str = trim(second).c_str();
            L.resize(numLayer);
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                L[i] = atoi(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "Sparsity")
        {
          string str = trim(second).c_str();
            Sparsity.resize(numLayer*2);
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                Sparsity[i] = atof(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "Batchsize")
        {
            Batchsize = atoi(trim(second).c_str());
        }
        else if (trim(first) == "Rehash")
        {
            Rehash = atoi(trim(second).c_str());
        }
        else if (trim(first) == "Rebuild")
        {
            Rebuild = atoi(trim(second).c_str());
        }
        else if (trim(first) == "InputDim")
        {
            InputDim = atoi(trim(second).c_str());
        }
        else if (trim(first) == "totRecords")
        {
            totRecords = atoi(trim(second).c_str());
        }
        else if (trim(first) == "totRecordsTest")
        {
            totRecordsTest = atoi(trim(second).c_str());
        }
        else if (trim(first) == "Epoch")
        {
            Epoch = atoi(trim(second).c_str());
        }
        else if (trim(first) == "Lr")
        {
            Lr = atof(trim(second).c_str());
        }
        else if (trim(first) == "Stepsize")
        {
            Stepsize = atoi(trim(second).c_str());
        }
        else if (trim(first) == "numLayer")
        {
            numLayer = atoi(trim(second).c_str());
        }
        else if (trim(first) == "logFile")
        {
            logFile = trim(second).c_str();
        }
        else if (trim(first) == "sizesOfLayers")
        {
            string str = trim(second).c_str();
            sizesOfLayers.resize(numLayer);
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                sizesOfLayers[i] = atoi(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "trainData")
        {
            trainData = trim(second).c_str();
        }
        else if (trim(first) == "testData")
        {
            testData = trim(second).c_str();
        }
        else if (trim(first) == "weight")
        {
            Weights = trim(second).c_str();
        }
        else if (trim(first) == "savedweight")
        {
            savedWeights = trim(second).c_str();
            cerr << "savedWeights=" << savedWeights << endl;
        }
        else
        {
            cout << "Error Parsing conf File at Line" << endl;
            cout << str << endl;
        }
    }
}

std::string PrintVec(const vector<string> &vec)
{
  stringstream strm;
  for (vector<string>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter) {
    strm << *iter << " ";
  }
  return strm.str();
}

void ReadHeader(const string &str, int &numLines, int &numInClass, int &numOutClass)
{
  vector<int> vec = Tokenize<int>(str);
  assert(vec.size() == 3);
  numLines = vec[0];
  numInClass = vec[1];
  numOutClass = vec[2];
  cerr << "header " << numLines << " " << numInClass << " " << numOutClass << endl;
}

void EvalDataSVM(int numBatchesTest,  Network &_mynet, int iter){
    cerr << "Start EvalDataSVM" << endl;
    int totCorrect = 0;
    int debugnumber = 0;
    std::ifstream testfile(testData);
    string str;
    //Skipe header
    std::getline( testfile, str );

    int numLines;
    int numInClass;
    int numOutClass;
    ReadHeader(str, numLines, numInClass, numOutClass);

    ofstream outputFile(logFile,  std::ios_base::app);
    for (int i = 0; i < numBatchesTest; i++) {
        vector<int*> records(Batchsize);
        vector<float*> values(Batchsize);
        vector<int> sizes(Batchsize);
        vector<int*> labels(Batchsize);
        vector<int> labelsize(Batchsize);
        int nonzeros = 0;
        int count = 0;
        while (std::getline(testfile, str)) {
            //cerr << "str=" << str << endl;
            char *mystring = &str[0];
            char *pch, *pchlabel;
            int track = 0;
            vector<string> list;
            vector<string> value;
            vector<string> label;
            pch = strtok(mystring, " ");
            pch = strtok(NULL, " :");
            while (pch != NULL) {
                if (track % 2 == 0)
                    list.push_back(pch);
                else if (track%2==1)
                    value.push_back(pch);
                track++;
                pch = strtok(NULL, " :");
            }

            pchlabel = strtok(mystring, ",");
            while (pchlabel != NULL) {
                label.push_back(pchlabel);
                pchlabel = strtok(NULL, ",");
            }

            //cerr << "   list=" << PrintVec(list) << endl;
            //cerr << "   value=" << PrintVec(value) << endl;
            //cerr << "   label=" << PrintVec(label) << endl;
           
            nonzeros += list.size();
            records[count] = new int[list.size()];
            values[count] = new float[list.size()];
            labels[count] = new int[label.size()];
            sizes[count] = list.size();
            labelsize[count] = label.size();

            int currcount = 0;
            debugnumber++;
            vector<string>::iterator it;
            for (it = list.begin(); it < list.end(); it++) {
                int inClass = stoi(*it);
                assert(inClass < numInClass);
                //cerr << "inClass=" << inClass << endl;
                records[count][currcount] = inClass;
                currcount++;
            }
            currcount = 0;
            for (it = value.begin(); it < value.end(); it++) {
                values[count][currcount] = stof(*it);
                currcount++;
            }
            currcount = 0;
            for (it = label.begin(); it < label.end(); it++) {
                int label = stoi(*it);
                assert(label < numOutClass);
                //cerr << "label=" << label << endl;
                labels[count][currcount] = label;
                currcount++;
            }

            count++;
            if (count >= Batchsize)
                break;
        }

        int num_features = 0, num_labels = 0;
        for (int i = 0; i < Batchsize; i++)
        {
            num_features += sizes[i];
            num_labels += labelsize[i];
        }

        std::cout << Batchsize << " records, with "<< num_features << " features and " << num_labels << " labels" << " debugnumber " << debugnumber << std::endl;
        int correctPredict = _mynet.predictClass(records, values, sizes, labels, labelsize, numInClass, numOutClass);
        totCorrect += correctPredict;
        std::cout <<" iter "<< i << ": " << totCorrect*1.0/(Batchsize*(i+1)) << " correct" << std::endl;

        for (int d = 0; d < Batchsize; d++) {
            delete[] records[d];
            delete[] values[d];
        }
    }
    testfile.close();
    cout << "over all " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;
    outputFile << iter << " " << globalTime/1000 << " " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;

    cerr << "Finished EvalDataSVM" << endl;
}

void ReadDataSVM(size_t numBatches,  Network &_mynet, int epoch){
    cerr << "Start ReadDataSVM" << endl;
    std::ifstream file(trainData);
    std::string str;
    //skipe header
    std::getline( file, str );

    int numLines;
    int numInClass;
    int numOutClass;
    ReadHeader(str, numLines, numInClass, numOutClass);

    for (size_t i = 0; i < numBatches; i++) {
        if( (i+epoch*numBatches)%Stepsize==0) {
            EvalDataSVM(20, _mynet, epoch*numBatches+i);
        }

        vector<int*> records(Batchsize);
        vector<float*> values(Batchsize);
        vector<int> sizes(Batchsize);
        vector<int*> labels(Batchsize);
        vector<int> labelsize(Batchsize);

        //int **records = new int *[Batchsize];
        //float **values = new float *[Batchsize];
        //int *sizes = new int[Batchsize];
        //int **labels = new int *[Batchsize];
        //int *labelsize = new int[Batchsize];
        int nonzeros = 0;
        int count = 0;
        vector<string> list;
        vector<string> value;
        vector<string> label;
        while (std::getline(file, str)) {
            //cerr << "str=" << str << endl;
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
                else if (track%2==1)
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
            records[count] = new int[list.size()];
            values[count] = new float[list.size()];
            labels[count] = new int[label.size()];
            sizes[count] = list.size();
            labelsize[count] = label.size();
            int currcount = 0;
            vector<string>::iterator it;
            for (it = list.begin(); it < list.end(); it++) {
                int inClass = stoi(*it);
                assert(inClass < numInClass);
                records[count][currcount] = inClass;
                currcount++;
            }
            currcount = 0;
            for (it = value.begin(); it < value.end(); it++) {
                values[count][currcount] = stof(*it);
                currcount++;
            }

            currcount = 0;
            for (it = label.begin(); it < label.end(); it++) {
                int label = stoi(*it);
                assert(label < numOutClass);
                //cerr << "label=" << label << endl;
                labels[count][currcount] = label;
                currcount++;
            }

            count++;
            if (count >= Batchsize)
                break;
        }

        bool rehash = false;
        bool rebuild = false;
        if ((epoch*numBatches+i)%(Rehash/Batchsize) == ((size_t)Rehash/Batchsize-1)){
            if(Mode==1 || Mode==4) {
                rehash = true;
            }
        }

        if ((epoch*numBatches+i)%(Rebuild/Batchsize) == ((size_t)Rehash/Batchsize-1)){
            if(Mode==1 || Mode==4) {
                rebuild = true;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        // logloss
        _mynet.ProcessInput(records, values, sizes, labels, labelsize, epoch * numBatches + i,
                                            rehash, rebuild);

        auto t2 = std::chrono::high_resolution_clock::now();

        int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        globalTime+= timeDiffInMiliseconds;

        for (int d = 0; d < Batchsize; d++) {
            delete[] records[d];
            delete[] values[d];
            delete[] labels[d];
        }
    }
    file.close();

    cerr << "Finished ReadDataSVM" << endl;
}


int main(int argc, char* argv[])
{
  //***********************************
  // Parse Config File
  //***********************************
  parseconfig(argv[1]);

    //***********************************
    // Initialize Network
    //***********************************
    int numBatches = totRecords/Batchsize;
    int numBatchesTest = totRecordsTest/Batchsize;
    std::vector<NodeType> layersTypes(numLayer);
    cerr << "totRecords=" << totRecords << " Batchsize=" << Batchsize << " numBatches=" << numBatches << endl;
    cerr << "totRecordsTest=" << totRecordsTest << " Batchsize=" << Batchsize << " numBatchesTest=" << numBatchesTest << endl;

    for (int i=0; i<numLayer-1; i++){
        layersTypes[i] = NodeType::ReLU;
    }
    layersTypes[numLayer-1] = NodeType::Softmax;

    cnpy::npz_t arr;
    if (LOADWEIGHT) {
        arr = cnpy::npz_load(Weights);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    {
      //Network a(sizesOfLayers, layersTypes, numLayer, Batchsize, Lr, InputDim, K, L, RangePow, Sparsity, arr);
    }

    Network _mynet(sizesOfLayers, layersTypes, numLayer, Batchsize, Lr, InputDim, K, L, RangePow, Sparsity, arr);
    auto t2 = std::chrono::high_resolution_clock::now();
    float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Network Initialization takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;

    //***********************************
    // Start Training
    //***********************************

    for (int e=0; e< Epoch; e++) {
        ofstream outputFile(logFile);
        std::cerr << "Epoch " << e << " " << logFile << endl;
        outputFile<<"Epoch "<<e<<endl;
        // train
        ReadDataSVM(numBatches, _mynet, e);

        // test
        if(e==Epoch-1) {
            EvalDataSVM(numBatchesTest, _mynet, (e+1)*numBatches);
        }else{
            EvalDataSVM(50, _mynet, (e+1)*numBatches);
        }
        _mynet.saveWeights(savedWeights);

    }

    return 0;

}
