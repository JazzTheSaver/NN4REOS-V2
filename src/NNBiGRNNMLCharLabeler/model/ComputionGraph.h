#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;
	const static int max_char_length = 4096;
	int rnn_layer_size;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	WindowBuilder _word_window;

	vector<GRNNBuilder> _rnn_lefts;
	vector<GRNNBuilder> _rnn_rights;

	vector<vector<ConcatNode> > _rnn_concats;

	AvgPoolNode _avg_word_pooling;
	MaxPoolNode _max_word_pooling;
	MinPoolNode _min_word_pooling;

	ConcatNode _word_pooling_concat;

	vector<LookupNode> _char_inputs;
	WindowBuilder _char_window;
	vector<UniNode> _char_hidden;

	AvgPoolNode _avg_char_pooling;
	MaxPoolNode _max_char_pooling;
	MinPoolNode _min_char_pooling;
	ConcatNode _char_pooling_concat;

	ConcatNode _concat;
	LinearNode _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int char_length, int rnn_layer_size) {
		_word_inputs.resize(sent_length);
		_word_window.resize(sent_length);
		_rnn_lefts.resize(rnn_layer_size);
		_rnn_rights.resize(rnn_layer_size);
		_rnn_concats.resize(rnn_layer_size);

		for (int i = 0; i < rnn_layer_size; i++) {
			_rnn_lefts[i].resize(sent_length);
			_rnn_rights[i].resize(sent_length);
			_rnn_concats[i].resize(sent_length);
		}

		_avg_word_pooling.setParam(sent_length);
		_max_word_pooling.setParam(sent_length);
		_min_word_pooling.setParam(sent_length);

		_char_inputs.resize(char_length);
		_char_window.resize(char_length);
		_char_hidden.resize(char_length);
		_avg_char_pooling.setParam(char_length);
		_max_char_pooling.setParam(char_length);
		_min_char_pooling.setParam(char_length);
	}

	inline void clear(){
		Graph::clear();
		_word_inputs.clear();
		_word_window.clear();
		_char_inputs.clear();
		_char_window.clear();
		_char_hidden.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		rnn_layer_size = opts.rnnLayerSize;
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb, mem);
		}
	
		_word_window.init(opts.wordDim, opts.wordContext, mem);
		for (int idx = 0; idx < rnn_layer_size; idx++) {
			_rnn_lefts[idx].init(&model.rnn_left_layers[idx], opts.dropProb, true, mem);
			_rnn_rights[idx].init(&model.rnn_right_layers[idx], opts.dropProb, false, mem);
			for (int idy = 0; idy < _rnn_concats[idx].size(); idy++)
				_rnn_concats[idx][idy].init(opts.rnnHiddenSize * 2, -1, mem);
		}

		_avg_word_pooling.init(opts.rnnHiddenSize * 2, -1, mem);
		_max_word_pooling.init(opts.rnnHiddenSize * 2, -1, mem);
		_min_word_pooling.init(opts.rnnHiddenSize * 2, -1, mem);
		_word_pooling_concat.init(opts.rnnHiddenSize * 2 * 3, -1, mem);

		for (int idx = 0; idx < _char_inputs.size(); idx++) {
			_char_inputs[idx].setParam(&model.chars);
			_char_inputs[idx].init(opts.charDim, opts.dropProb, mem);
			_char_hidden[idx].setParam(&model.char_hidden_linear);
			_char_hidden[idx].init(opts.charHiddenSize, opts.dropProb, mem);
		}

		_char_window.init(opts.charDim, opts.charContext, mem);
		_avg_char_pooling.init(opts.charHiddenSize, -1, mem);
		_max_char_pooling.init(opts.charHiddenSize, -1, mem);
		_min_char_pooling.init(opts.charHiddenSize, -1, mem);
		_char_pooling_concat.init(opts.charHiddenSize * 3, -1, mem);

		_concat.init(opts.charHiddenSize * 3 + opts.rnnHiddenSize  * 2 * 3, -1, mem);
		_output.setParam(&model.olayer_linear);
		_output.init(opts.labelSize, -1, mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		// second step: build graph
		//forward
		int words_num = feature.m_tweet_words.size();
		if (words_num > max_sentence_length)
			words_num = max_sentence_length;
		for (int idx = 0; idx < words_num; idx++) {
			_word_inputs[idx].forward(this, feature.m_tweet_words[idx]);
		}

		_word_window.forward(this, getPNodes(_word_inputs, words_num));

		_rnn_lefts[0].forward(this, getPNodes(_word_window._outputs, words_num));
		_rnn_rights[0].forward(this, getPNodes(_word_window._outputs, words_num));
		for (int i = 0; i < words_num; i++)
			_rnn_concats[0][i].forward(this, &_rnn_lefts[0]._output[i], &_rnn_rights[0]._output[i]);

		for (int idx = 1; idx < rnn_layer_size; idx++) {
			_rnn_lefts[idx].forward(this, getPNodes(_rnn_concats[idx - 1], words_num));
			_rnn_rights[idx].forward(this, getPNodes(_rnn_concats[idx - 1], words_num));
			for (int i = 0; i < words_num; i++)
				_rnn_concats[idx][i].forward(this, &_rnn_lefts[idx]._output[i], &_rnn_rights[idx]._output[i]);
		}

		_avg_word_pooling.forward(this, getPNodes(_rnn_concats[rnn_layer_size - 1], words_num));
		_max_word_pooling.forward(this, getPNodes(_rnn_concats[rnn_layer_size - 1], words_num));
		_min_word_pooling.forward(this, getPNodes(_rnn_concats[rnn_layer_size - 1], words_num));
		_word_pooling_concat.forward(this, &_avg_word_pooling, &_max_word_pooling, &_min_word_pooling);

		int chars_num = feature.m_chars.size();
		if (chars_num > max_char_length)
			chars_num = max_char_length;
		for (int i = 0; i < chars_num; i++) {
			_char_inputs[i].forward(this, feature.m_chars[i]);
		}
		_char_window.forward(this, getPNodes(_char_inputs, chars_num));

		for (int i = 0; i < chars_num; i++) {
			_char_hidden[i].forward(this, &_char_window._outputs[i]);
		}
		_avg_char_pooling.forward(this, getPNodes(_char_hidden, chars_num));
		_max_char_pooling.forward(this, getPNodes(_char_hidden, chars_num));
		_min_char_pooling.forward(this, getPNodes(_char_hidden, chars_num));
		_char_pooling_concat.forward(this, &_avg_char_pooling, &_max_char_pooling, &_min_char_pooling);

		_concat.forward(this, &_word_pooling_concat, &_char_pooling_concat);
		_output.forward(this, &_concat);
	}
};

#endif /* SRC_ComputionGraph_H_ */