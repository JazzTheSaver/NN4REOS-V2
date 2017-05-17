#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	vector<UniParams> hidden_linears;
	UniParams olayer_linear; // output
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;
		opts.windowHiddenOutput = opts.hiddenSize * opts.wordWindow;
		opts.labelSize = labelAlpha.size();

		hidden_linears.resize(opts.cnnLayerSize);
		hidden_linears[0].initial(opts.hiddenSize, opts.windowOutput, true, mem);
		for(int idx = 1; idx < opts.cnnLayerSize;idx++)
			hidden_linears[idx].initial(opts.hiddenSize, opts.windowOutput, true, mem);

		opts.inputSize = opts.hiddenSize * 3;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		return true;
	}

	bool TestInitial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;
		opts.labelSize = labelAlpha.size();
		opts.inputSize = opts.hiddenSize * 3;
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		int cnn_layer_size = hidden_linears.size();
		for(int idx = 0; idx < cnn_layer_size; idx++)
			hidden_linears[idx].exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	// will add it later
	void saveModel(std::ofstream &os) const{
		wordAlpha.write(os);
		words.save(os);
		//hidden_linear.save(os);
		olayer_linear.save(os);
		labelAlpha.write(os);
	}

	void loadModel(std::ifstream &is, AlignedMemoryPool* mem = NULL){
		wordAlpha.read(is);
		words.load(is, &wordAlpha, mem);
		//hidden_linear.load(is, mem);
		olayer_linear.load(is, mem);
		labelAlpha.read(is);
	}

};

#endif /* SRC_ModelParams_H_ */