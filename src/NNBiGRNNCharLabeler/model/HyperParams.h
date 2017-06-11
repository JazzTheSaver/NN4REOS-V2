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

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


	void saveModel(std::ofstream &os) const{
		os << nnRegular << endl; 
		os << adaAlpha << endl; 
		os << adaEps << endl; 

		os << rnnHiddenSize << endl;
		os << wordContext << endl;
		os << wordWindow << endl;

		os << charHiddenSize << endl;
		os << charContext << endl;
		os << charWindow << endl;

		os << wordWindowOutput << endl;
		os << charWindowOutput << endl;
		os << dropProb << endl;

		os << wordDim << endl;
		os << charDim << endl;
		os << inputSize << endl;
		os << labelSize << endl;
	}

	void loadModel(std::ifstream &is){
		is >> nnRegular; 
		is >> adaAlpha; 
		is >> adaEps; 

		is >> rnnHiddenSize;
		is >> wordContext;
		is >> wordWindow;

		is >> charHiddenSize;
		is >> charContext;
		is >> charWindow;

		is >> wordWindowOutput;
		is >> charWindowOutput;
		is >> dropProb;

		is >> wordDim;
		is >> charDim;
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