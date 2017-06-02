#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization

	int hiddenSize;
	int cnnLayerSize;
	int wordContext;
	int wordWindow;

	int charHiddenSize;
	int charContext;
	int charWindow;

	int wordWindowOutput;
	int charWindowOutput;
	int windowHiddenOutput;
	dtype dropProb;


	//auto generated
	int wordDim;
	int charDim;
	int inputSize;
	int labelSize;

public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		if (opt.cnnLayerSize < 1)
			cnnLayerSize = 1;
		else
			cnnLayerSize = opt.cnnLayerSize;
		hiddenSize = opt.hiddenSize;
		wordContext = opt.wordcontext;
		charContext = opt.charcontext;
		charHiddenSize = opt.charhiddenSize;
		dropProb = opt.dropProb;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


	void saveModel(std::ofstream &os) const{
		os << nnRegular << std::endl;
		os << adaAlpha << std::endl;
		os << adaEps << std::endl;

		os << hiddenSize << std::endl;
		os << wordContext << std::endl;
		os << wordWindow << std::endl;
		os << wordWindowOutput << std::endl;
		os << dropProb << std::endl;


		os << wordDim << std::endl;
		os << inputSize << std::endl;
		os << labelSize << std::endl;
	}

	void loadModel(std::ifstream &is){
		is >> nnRegular;
		is >> adaAlpha;
		is >> adaEps;

		is >> hiddenSize;
		is >> wordContext;
		is >> wordWindow;
		is >> wordWindowOutput;
		is >> dropProb;


		is >> wordDim;
		is >> inputSize;
		is >> labelSize;

		bAssigned = true;
	}
public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */