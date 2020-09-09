#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <string>
#include <random>

#include <betterthanmnist.h>

using namespace std;

const unsigned c_numInputNeurons = 784;
const unsigned c_numHiddenNeurons = 30;
const unsigned c_numOutputNeurons = 10;

const unsigned c_trainingGenerations = 30;
const unsigned c_batchSize = 10;
const float c_learningRate = 3.0f;
const unsigned c_numTestsForGen = 300;

const string c_init_filename = "init90.txt";


class NeuralNetwork
{
public:
    NeuralNetwork (unsigned b_size = c_batchSize, float l_rate = c_learningRate){
        my_batch_size = b_size;
        my_learning_rate = l_rate;

        random_device r;
        normal_distribution<float> dist(0, 1);

        for (float& f : hiddenLayerBiases)
            f = dist(r);

        for (float& f : outputLayerBiases)
            f = dist(r);

        for (float& f : hiddenLayerWeights)
            f = dist(r);

        for (float& f : outputLayerWeights)
            f = dist(r);
        cout<<"[+] Network created."<<endl;
    }

    void fill_arr(float* arr,unsigned arr_size, float fill){
        for (unsigned i = 0;i < arr_size; ++i)
            arr[i] = fill;
    }

    void Train (BetterThanMnist& trainingData){
        unsigned trainingIndex = 0;
        while (trainingIndex < c_numTestsForGen){

            fill_arr(batchHiddenLayerBiasesDeltaDer, c_numHiddenNeurons, 0.0f);
            fill_arr(batchOutputLayerBiasesDeltaDer, c_numOutputNeurons, 0.0f);
            fill_arr(batchHiddenLayerWeightsDeltaDer, c_numInputNeurons*c_numHiddenNeurons, 0.0f);
            fill_arr(batchOutputLayerWeightsDeltaDer, c_numHiddenNeurons*c_numOutputNeurons, 0.0f);

            unsigned batchIndex = 0;
            while (batchIndex < my_batch_size && trainingIndex < c_numTestsForGen){

                unsigned imageLabel = 0;
                vector<float> pixels = trainingData.GetImage(imageLabel);

                ForwardPass(pixels);

                BackwardPass(pixels, imageLabel);

                // sum derivatives
                for (unsigned i = 0; i < c_numHiddenNeurons; ++i)
                    batchHiddenLayerBiasesDeltaDer[i] += hiddenLayerBiasesDeltaDerGradient[i];
                for (unsigned i = 0; i < c_numOutputNeurons; ++i)
                    batchOutputLayerBiasesDeltaDer[i] += outputLayerBiasesDeltaDerGradient[i];
                for (unsigned i = 0; i < c_numInputNeurons*c_numHiddenNeurons; ++i)
                    batchHiddenLayerWeightsDeltaDer[i] += hiddenLayerWeightsDeltaDer[i];
                for (unsigned i = 0; i < c_numHiddenNeurons*c_numOutputNeurons; ++i)
                    batchOutputLayerWeightsDeltaDer[i] += outputLayerWeightsDeltaDer[i];

                ++trainingIndex;
                ++batchIndex;
            }

            float batchLearningRate = my_learning_rate / float(batchIndex);

            // apply training
            for (unsigned i = 0; i < c_numHiddenNeurons; ++i)
                hiddenLayerBiases[i] -= batchHiddenLayerBiasesDeltaDer[i] * batchLearningRate;
            for (unsigned i = 0; i < c_numOutputNeurons; ++i)
                outputLayerBiases[i] -= batchOutputLayerBiasesDeltaDer[i] * batchLearningRate;
            for (unsigned i = 0; i < c_numInputNeurons*c_numHiddenNeurons; ++i)
                hiddenLayerWeights[i] -= batchHiddenLayerWeightsDeltaDer[i] * batchLearningRate;
            for (unsigned i = 0; i < c_numHiddenNeurons*c_numOutputNeurons; ++i)
                outputLayerWeights[i] -= batchOutputLayerWeightsDeltaDer[i] * batchLearningRate;
        }
    }

    // input pixels -> answer
    unsigned ForwardPass (vector<float>& pixels){
        // hidden layer
        for (unsigned neuronIndex = 0; neuronIndex < c_numHiddenNeurons; ++neuronIndex){
            float sum = hiddenLayerBiases[neuronIndex];

            for (unsigned inputIndex = 0; inputIndex < c_numInputNeurons; ++inputIndex)
                sum += pixels[inputIndex] * hiddenLayerWeights[HiddenLayerWeightIndex(inputIndex, neuronIndex)];

            hiddenLayerVals[neuronIndex] = 1.0f / (1.0f + exp(-sum));
        }

        // output layer
        for (unsigned neuronIndex = 0; neuronIndex < c_numOutputNeurons; ++neuronIndex){
            float sum = outputLayerBiases[neuronIndex];

            for (unsigned inputIndex = 0; inputIndex < c_numHiddenNeurons; ++inputIndex)
               sum += hiddenLayerVals[inputIndex] * outputLayerWeights[OutputLayerWeightIndex(inputIndex, neuronIndex)];

            outputLayerVals[neuronIndex] = 1.0f / (1.0f + exp(-sum));
        }

        // return max value
        float maxOutput = outputLayerVals[0];
        unsigned maxLabel = 0;
        for (unsigned neuronIndex = 1; neuronIndex < c_numOutputNeurons; ++neuronIndex){
            if (outputLayerVals[neuronIndex] > maxOutput){
                maxOutput = outputLayerVals[neuronIndex];
                maxLabel = neuronIndex;
            }
        }
        return maxLabel;
    }

    const float* GetHiddenLayerBiases () const { return hiddenLayerBiases; }
    const float* GetOutputLayerBiases () const { return outputLayerBiases; }
    const float* GetHiddenLayerWeights () const { return hiddenLayerWeights; }
    const float* GetOutputLayerWeights () const { return outputLayerWeights; }

    void initialize(){
        fstream fin;
        fin.open(c_init_filename);
        for (float& f : hiddenLayerBiases)
            fin >> f;
        for (float& f : outputLayerBiases)
            fin >> f;
        for (float& f : hiddenLayerWeights)
            fin >> f;
        for (float& f : outputLayerWeights)
            fin >> f;
        cout<<"[+] Neural Network iniitalized."<<endl;
    }

private:

    unsigned HiddenLayerWeightIndex (unsigned inputIndex, unsigned hiddenLayerNeuronIndex){
        return hiddenLayerNeuronIndex * c_numInputNeurons + inputIndex;
    }

    unsigned OutputLayerWeightIndex (unsigned hiddenLayerNeuronIndex, unsigned outputLayerNeuronIndex){
        return outputLayerNeuronIndex * c_numHiddenNeurons + hiddenLayerNeuronIndex;
    }

    void BackwardPass (vector<float>& pixels, unsigned correctLabel){
        // output layer
        for (unsigned neuronIndex = 0; neuronIndex < c_numOutputNeurons; ++neuronIndex){
            float bestOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;

            float deltaError = outputLayerVals[neuronIndex] - bestOutput;
            float delta0 = outputLayerVals[neuronIndex] * (1.0f - outputLayerVals[neuronIndex]);

            outputLayerBiasesDeltaDerGradient[neuronIndex] = deltaError * delta0;

            for (unsigned inputIndex = 0; inputIndex < c_numHiddenNeurons; ++inputIndex)
                outputLayerWeightsDeltaDer[OutputLayerWeightIndex(inputIndex, neuronIndex)] = outputLayerBiasesDeltaDerGradient[neuronIndex] * hiddenLayerVals[inputIndex];
        }

        // hidden layer
        for (unsigned neuronIndex = 0; neuronIndex < c_numHiddenNeurons; ++neuronIndex){

            float deltaErrorSum = 0.0f;
            for (unsigned destinationNeuronIndex = 0; destinationNeuronIndex < c_numOutputNeurons; ++destinationNeuronIndex)
                deltaErrorSum += outputLayerBiasesDeltaDerGradient[destinationNeuronIndex] * outputLayerWeights[OutputLayerWeightIndex(neuronIndex, destinationNeuronIndex)];
            float delta0 = hiddenLayerVals[neuronIndex] * (1.0f - hiddenLayerVals[neuronIndex]);
            hiddenLayerBiasesDeltaDerGradient[neuronIndex] = deltaErrorSum * delta0;

            for (unsigned inputIndex = 0; inputIndex < c_numInputNeurons; ++inputIndex)
                hiddenLayerWeightsDeltaDer[HiddenLayerWeightIndex(inputIndex, neuronIndex)] = hiddenLayerBiasesDeltaDerGradient[neuronIndex] * pixels[inputIndex];
        }
    }

private:
    unsigned my_batch_size;
    float my_learning_rate;
    // biases and weights
    float   hiddenLayerBiases[c_numHiddenNeurons];
    float   outputLayerBiases[c_numOutputNeurons];

    float   hiddenLayerWeights[c_numInputNeurons*c_numHiddenNeurons];
    float   outputLayerWeights[c_numHiddenNeurons*c_numOutputNeurons];

    // neuron activation values
    float   hiddenLayerVals[c_numHiddenNeurons];
    float   outputLayerVals[c_numOutputNeurons];

    // derivatives of biases and weights for every test. GRADIENT
    float   hiddenLayerBiasesDeltaDerGradient[c_numHiddenNeurons];
    float   outputLayerBiasesDeltaDerGradient[c_numOutputNeurons];

    float   hiddenLayerWeightsDeltaDer[c_numInputNeurons*c_numHiddenNeurons];
    float   outputLayerWeightsDeltaDer[c_numHiddenNeurons*c_numOutputNeurons];

    // derivatives of biases and weights for every batch. Average of all items in batch.
    float   batchHiddenLayerBiasesDeltaDer[c_numHiddenNeurons];
    float   batchOutputLayerBiasesDeltaDer[c_numOutputNeurons];

    float   batchHiddenLayerWeightsDeltaDer[c_numInputNeurons*c_numHiddenNeurons];
    float   batchOutputLayerWeightsDeltaDer[c_numHiddenNeurons*c_numOutputNeurons];
};

#endif // NEURALNETWORK_H
