# Internship
Named Entity Recognition for plant gene dataset using Bi-LSTM + CNN + CRF

### Input:
 Input of model consist two main parts:
 * Character representation: extracted from a Convolutional Neural Network with different kernel size. It performs a narrow convolution to extract information of character in words.
 * Word representation: extracted by embedding through a pre-trained model. In this repository, we used GloVe word embedding model to extract information of words. 

### Model: 
Main model consists two main stage: 
 * CNN model: a CNN network with different kernel size work at characater level to extract information from words. 
 * bi-lSTM model: includings independent LSTM model with same size but in opposite direction.
 * CRF layer: output from bi-LSTM will be used as input for CRF to compute the tag score and also predict the word is entity or not.

### References
 * https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
 * http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
