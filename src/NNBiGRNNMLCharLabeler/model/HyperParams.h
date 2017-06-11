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

	int rnnHiddenSize;
	int wordContext;
	int wordWindow;

	int charHiddenSize;
	int charContext;
	int charWindow;

	int wordWindowOutput;
	int charWindowOutput;
	dtype dropProb;


	//auto generated
	int wordDim;
	int charDim;
	int inputSize;
	int labelSize;
	int rnnLayerSize;

public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		rnnHiddenSize = opt.rnnHiddenSize;
		charHiddenSize = opt.charhiddenSize;
		wordContext = opt.wordcontext;
		charContext = opt.charcontext;
		dropProb = opt.dropProb;
		rnnLayerSize = opt.rnnLayerSize;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


	void saveModel(std::ofstream &os) const{
	}

	void loadModel(std::ifstream &is){
	}
public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */