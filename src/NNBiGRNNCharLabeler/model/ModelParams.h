#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet charAlpha; // should be initialized outside
	LookupTable chars; // should be initialized outside
	GRNNParams rnn_left_layer;
	GRNNParams rnn_right_layer;
	UniParams char_hidden_linear;
	UniParams olayer_linear; // output
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL) {

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0) {
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.wordWindowOutput = opts.wordDim * opts.wordWindow;

		opts.charDim = chars.nDim;
		opts.charWindow = opts.charContext * 2 + 1;
		opts.charWindowOutput = opts.charDim * opts.charWindow;

		opts.labelSize = labelAlpha.size();
		rnn_left_layer.initial(opts.rnnHiddenSize, opts.wordWindowOutput, mem);
		rnn_right_layer.initial(opts.rnnHiddenSize, opts.wordWindowOutput, mem);
		char_hidden_linear.initial(opts.charHiddenSize, opts.charWindowOutput, true, mem);
		opts.inputSize = opts.rnnHiddenSize * 2 * 3 + opts.charHiddenSize * 3;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		return true;
	}

	bool TestInitial(HyperParams& opts, AlignedMemoryPool* mem = NULL) {

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0) {
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

	void exportModelParams(ModelUpdate& ada) {
		words.exportAdaParams(ada);
		chars.exportAdaParams(ada);
		rnn_left_layer.exportAdaParams(ada);
		rnn_right_layer.exportAdaParams(ada);
		char_hidden_linear.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad) {
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	// will add it later
	void saveModel(std::ofstream &os) const {
		wordAlpha.write(os);
		words.save(os);
		charAlpha.write(os);
		chars.save(os);
		rnn_left_layer.save(os);
		rnn_right_layer.save(os);
		char_hidden_linear.save(os);
		olayer_linear.save(os);
		labelAlpha.write(os);
	}

	void loadModel(std::ifstream &is, AlignedMemoryPool* mem = NULL) {
		wordAlpha.read(is);
		words.load(is, &wordAlpha, mem);
		charAlpha.read(is);
		chars.load(is, &charAlpha, mem);
		rnn_left_layer.load(is, mem);
		rnn_right_layer.load(is, mem);
		char_hidden_linear.load(is, mem);
		olayer_linear.load(is, mem);
		labelAlpha.read(is);
	}

};

#endif /* SRC_ModelParams_H_ */