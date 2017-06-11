#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet charAlpha; // should be initialized outside
	LookupTable chars; // should be initialized outside

	vector<GRNNParams> rnn_left_layers;
	vector<GRNNParams> rnn_right_layers;

	UniParams char_hidden_linear;
	UniParams olayer_linear; // output
	int rnn_layer_size;
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		rnn_layer_size = opts.rnnLayerSize;
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.wordWindowOutput = opts.wordDim * opts.wordWindow;

		opts.charDim = chars.nDim;
		opts.charWindow = opts.charContext * 2 + 1;
		opts.charWindowOutput = opts.charDim * opts.charWindow;

		opts.labelSize = labelAlpha.size();

		rnn_left_layers.resize(rnn_layer_size);
		rnn_right_layers.resize(rnn_layer_size);
		
		rnn_left_layers[0].initial(opts.rnnHiddenSize, opts.wordWindowOutput, mem);
		rnn_right_layers[0].initial(opts.rnnHiddenSize, opts.wordWindowOutput, mem);

		for (int idx = 1; idx < rnn_layer_size; idx++) {
			rnn_left_layers[idx].initial(opts.rnnHiddenSize, opts.rnnHiddenSize * 2, mem);
			rnn_right_layers[idx].initial(opts.rnnHiddenSize, opts.rnnHiddenSize * 2, mem);
		}

		char_hidden_linear.initial(opts.charHiddenSize, opts.charWindowOutput, true, mem);
		opts.inputSize = opts.rnnHiddenSize * 2 * 3 + opts.charHiddenSize * 3;
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
		opts.wordWindowOutput = opts.wordDim * opts.wordWindow;

		opts.charDim = chars.nDim;
		opts.charWindow = opts.charContext * 2 + 1;
		opts.charWindowOutput = opts.charDim * opts.charWindow;

		opts.labelSize = labelAlpha.size();
		opts.inputSize = opts.rnnHiddenSize * 2 * 3 + opts.charHiddenSize * 3;
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		chars.exportAdaParams(ada);
		for (int i = 0; i < rnn_layer_size; i++) {
			rnn_left_layers[i].exportAdaParams(ada);
			rnn_right_layers[i].exportAdaParams(ada);
		}
		char_hidden_linear.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	// will add it later
	void saveModel(std::ofstream &os) const{
	}

	void loadModel(std::ifstream &is, AlignedMemoryPool* mem = NULL){
	}

};

#endif /* SRC_ModelParams_H_ */