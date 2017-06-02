#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;
	const static int max_char_length = 4096;

public:
	// node instances
	vector<LookupNode> _word_inputs;

	int cnn_layer_size;
	vector<WindowBuilder> _word_windows;
	vector<vector<UniNode> > _hiddens;

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
	inline void createNodes(int sent_length, int char_length, int layer_size){
		cnn_layer_size = layer_size;
		_word_inputs.resize(sent_length);

		_word_windows.resize(cnn_layer_size);
		_hiddens.resize(cnn_layer_size);

		cnn_layer_size = layer_size;
		for (int idx = 0; idx < cnn_layer_size; idx++) {
			_word_windows[idx].resize(sent_length);
			_hiddens[idx].resize(sent_length);
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

		for(int idx = 0; idx < cnn_layer_size; idx++){
			_word_windows[idx].clear();
			_hiddens[idx].clear();
		}

		_word_windows.clear();
		_hiddens.clear();
		_char_inputs.clear();
		_char_window.clear();
		_char_hidden.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb, mem);

			for (int idy = 0; idy < cnn_layer_size; idy++) {
				_hiddens[idy][idx].setParam(&model.hidden_linears[idy]);
				_hiddens[idy][idx].init(opts.hiddenSize, opts.dropProb, mem);
			}

		}

		_word_windows[0].init(opts.wordDim, opts.wordContext, mem);
		for (int idy = 1; idy < cnn_layer_size; idy++) {
			_word_windows[idy].init(opts.hiddenSize, opts.wordContext, mem);
		}

		_avg_word_pooling.init(opts.hiddenSize, -1, mem);
		_max_word_pooling.init(opts.hiddenSize, -1, mem);
		_min_word_pooling.init(opts.hiddenSize, -1, mem);
		_word_pooling_concat.init(opts.hiddenSize * 3, -1, mem);

		_output.setParam(&model.olayer_linear);
		_output.init(opts.labelSize, -1, mem);

		for(int idx = 0; idx < _char_inputs.size(); idx++) {
			_char_inputs[idx].setParam(&model.chars);
			_char_inputs[idx].init(opts.charDim, opts.dropProb, mem);
		}
		_char_window.init(opts.charDim, opts.charContext, mem);
		for(int idx = 0; idx < _char_hidden.size(); idx++){
			_char_hidden[idx].setParam(&model.char_hidden_linear);
			_char_hidden[idx].init(opts.charHiddenSize, opts.dropProb, mem);
		}
		_avg_char_pooling.init(opts.charHiddenSize, opts.dropProb, mem);
		_max_char_pooling.init(opts.charHiddenSize, opts.dropProb, mem);
		_min_char_pooling.init(opts.charHiddenSize, opts.dropProb, mem);
		_char_pooling_concat.init(opts.charHiddenSize * 3, -1, mem);

		_concat.init(3 * (opts.charHiddenSize + opts.hiddenSize), -1, mem);

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
		for (int i = 0; i < words_num; i++) {
			_word_inputs[i].forward(this, feature.m_tweet_words[i]);
		}
		_word_windows[0].forward(this, getPNodes(_word_inputs, words_num));

		for (int i = 0; i < words_num; i++) {
			_hiddens[0][i].forward(this, &_word_windows[0]._outputs[i]);
		}

		for(int i = 1; i < cnn_layer_size; i++){
			_word_windows[i].forward(this, getPNodes(_hiddens[i - 1], words_num));
			for (int j = 0; j < words_num; j++) {
				_hiddens[i][j].forward(this, &_word_windows[i]._outputs[j]);
			}
		}

		_avg_word_pooling.forward(this, getPNodes(_hiddens[cnn_layer_size - 1], words_num));
		_max_word_pooling.forward(this, getPNodes(_hiddens[cnn_layer_size - 1], words_num));
		_min_word_pooling.forward(this, getPNodes(_hiddens[cnn_layer_size - 1], words_num));

		_word_pooling_concat.forward(this, &_avg_word_pooling, &_max_word_pooling, &_min_word_pooling);
		
		int chars_num = feature.m_chars.size();
		if (chars_num > max_char_length)
			chars_num = max_char_length;

		for (int i = 0; i < chars_num; i++)
			_char_inputs[i].forward(this, feature.m_chars[i]);
		_char_window.forward(this, getPNodes(_char_inputs, chars_num));
		for (int i = 0; i < chars_num; i++)
			_char_hidden[i].forward(this, &_char_window._outputs[i]);
		_avg_char_pooling.forward(this, getPNodes(_char_hidden, chars_num));
		_max_char_pooling.forward(this, getPNodes(_char_hidden, chars_num));
		_min_char_pooling.forward(this, getPNodes(_char_hidden, chars_num));

		_char_pooling_concat.forward(this, &_avg_char_pooling, &_max_char_pooling, &_min_char_pooling);

		_concat.forward(this, &_word_pooling_concat, &_char_pooling_concat);

		_output.forward(this, &_concat);
	}
};

#endif /* SRC_ComputionGraph_H_ */