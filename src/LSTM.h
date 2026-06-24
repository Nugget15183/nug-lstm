
#ifndef LSTM_LSTM_H
#define LSTM_LSTM_H
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include "matrix.hpp"

using namespace std;

inline default_random_engine re;

enum AType {
    TANH = 1,
    SIGMOID = 2,
};

struct weight {
    double value;
};

class Neuron {
public:
    vector<weight> weights;
    double bias = 0.0;

    double lastZ = 0.0;

    double lastOutput;
    vector<double> lastInput;

    AType atype = SIGMOID;

    Neuron(int inputSize) {
        weights = vector<weight>(inputSize);

        for (int i=0; i<weights.size(); i++) {
            uniform_real_distribution<double> unif(-1.0,1.0);
            weights[i].value = unif(re);
        }
    }

    double activate(vector<double> input) {
        double z = bias;
        double result = 0.0;

        if (weights.size() != input.size()) {
            throw runtime_error("wrong weight input size used, w: " + to_string(weights.size()) + " vs i: " + to_string(input.size()));
        }

        for (int i=0; i < input.size(); i++) {
            z += input[i] * weights[i].value;
        }

        if (atype == TANH) {
            result = tanh(z);
        } else if (atype == SIGMOID) {
            result = 1 / (1 + exp(-z));
        }

        lastInput = input;
        lastZ = z;
        lastOutput = result;

        return result;
    };
};

class Cell {
public:
    Cell(int gateSize, int InputSize) {
        inputSize = InputSize;
        size = gateSize;

        W_Out = feng::matrix<double>(inputSize, size);

        for (int x=0; x < inputSize; x++) {
            for (int y=0; y < size; y++) {
                uniform_real_distribution<double> unif(-1.0,1.0);
                W_Out[x][y] = unif(re);//initialize with small values
            }
        }

        for (int i=0; i < size; i++) {
            Neuron f = Neuron(1 + size);
            Neuron o = Neuron(1 + size);
            Neuron s = Neuron(1 + size);
            Neuron in = Neuron(1 + size);

            s.atype = TANH;

            Forget.push_back(f);
            Output.push_back(o);
            State.push_back(s);
            Input.push_back(in);
        }
    }

    int inputSize = 0;
    int size = 0;

    feng::matrix<double> W_Out;

    vector<Neuron> Forget;
    vector<Neuron> Output;
    vector<Neuron> State;
    vector<Neuron> Input;

    int t = 0;
    vector<double> C = vector<double>(size);
    vector<double> H = vector<double>(size);

    vector<vector<double>> h_hist;//old H's
    vector<vector<double>> c_hist;//old C's
};

class LSTM {
public:
    void train(vector<vector<int>> inputs, vector<vector<int>> targets, int epochs);
    void activate(vector<int> input);
private:
    Cell c = Cell(32, 27);
};


#endif //LSTM_LSTM_H