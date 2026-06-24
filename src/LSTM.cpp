#include "LSTM.h"
#include "matrix.hpp"

double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

void LSTM::train(vector<vector<int>> inputs, vector<vector<int>> targets, int epochs) {
    double learningRate = 0.01;
    for (int e=0; e<epochs; e++) {
        c.h_hist.clear();
        c.c_hist.clear();
        double totalLoss = 0.0;

        for (int i=0; i<inputs.size(); i++) {//for i in each example
           c.t++;

           //reset states
           c.H = vector<double>(c.size);
           c.C = vector<double>(c.size);

           vector<double> prevH = vector<double>(c.size, 0.0);
           vector<double> prevC = vector<double>(c.size, 0.0);

           vector<int> example = inputs[i];

           //forward pass
           for (int k=0; k<example.size(); k++) {//timestep loop (loop through each letter)
               int curToken = example[k];

               vector<double> input;
               input.push_back(curToken);

               for (int h=0; h<prevH.size(); h++) {
                   input.push_back(prevH[h]);
               }

               vector<double> f = vector<double>(c.size, 0.0);
               vector<double> it = vector<double>(c.size, 0.0);
               vector<double> g = vector<double>(c.size, 0.0);
               vector<double> o = vector<double>(c.size, 0.0);

               for (int i=0; i<c.Forget.size(); i++) {
                   f[i] = c.Forget[i].activate(input);
               }

               for (int k=0; k<c.Input.size(); k++) {
                   it[k] += c.Input[k].activate(input);
               }

               for (int i=0; i<c.State.size(); i++) {
                   g[i] += c.State[i].activate(input);
               }

               for (int i=0; i<c.Output.size(); i++) {
                   o[i] += c.Output[i].activate(input);
               }

               vector<double> C_t = vector<double>(c.size, 0.0);
               vector<double> H_t = vector<double>(c.size, 0.0);
               for (int j=0; j<c.size; j++) {
                   C_t[j] = (f[j] * prevC[j]) + (it[j] * g[j]);
                   H_t[j] = o[j] * tanh(C_t[j]);
               }

               prevC = C_t;
               prevH = H_t;

               //output layer
               vector<double> logits = vector<double>(c.inputSize, 0.0);

               for (int x=0; x < c.inputSize; x++) {
                   double sum = 0;
                   for (int y = 0; y < c.size; y++) {
                       sum += c.W_Out[x][y] * prevH[y];
                   }
                   logits[x] = sum;
               }

               vector<double> probs = vector<double>(c.inputSize);

               double maxLogit = logits[0];
               for (int b=0; b < logits.size(); b++) {
                   if (maxLogit < logits[b]) {
                       maxLogit = logits[b];
                   }
               }

               double sumExp = 0.0;
               for (int v=0; v < logits.size(); v++) {
                   sumExp += exp(logits[v] - maxLogit);
               }
               for (int v=0; v < logits.size(); v++) {
                   probs[v] = exp(logits[v] - maxLogit) / sumExp;
               }

               int target = targets[i][k];
               double loss = -log(probs[target]);
               totalLoss += loss;

               //backprop
               vector<double> dlogits = vector<double>(probs.size(), 0.0);

               feng::matrix<double> dW_Out(c.inputSize, c.size);

               vector<double> dH = vector<double>(c.size, 0.0);
               vector<double> dC_t = vector<double>(c.size, 0.0);
               vector<double> dPrevC = vector<double>(c.size, 0.0);

               vector<double> dF = vector<double>(c.size, 0.0);
               vector<double> dIt = vector<double>(c.size, 0.0);
               vector<double> dG = vector<double>(c.size, 0.0);
               vector<double> d0 = vector<double>(c.size, 0.0);

               //how much each neuron contributed to the loss
               vector<double> dZ_f = vector<double>(c.size, 0.0);
               vector<double> dZ_it = vector<double>(c.size, 0.0);
               vector<double> dZ_g = vector<double>(c.size, 0.0);
               vector<double> dZ_o = vector<double>(c.size, 0.0);

               vector<vector<double>> dW_Forget = vector<vector<double>>(c.size, vector<double>(1 + c.size, 0.00));
               vector<double> dBias_Forget = vector<double>(c.size, 0.0);

               vector<vector<double>> dW_Input = vector<vector<double>>(c.size, vector<double>(1 + c.size, 0.00));
               vector<double> dBias_Input = vector<double>(c.size, 0.0);

               vector<vector<double>> dW_Output = vector<vector<double>>(c.size, vector<double>(1 + c.size, 0.00));
               vector<double> dBias_Output = vector<double>(c.size, 0.0);

               vector<vector<double>> dW_State = vector<vector<double>>(c.size, vector<double>(1 + c.size, 0.00));
               vector<double> dBias_State = vector<double>(c.size, 0.0);

               for (int v=0; v < probs.size(); v++) {
                   dlogits[v] = probs[v];
               }
               dlogits[target] -= 1;

               for (int x=0; x < c.inputSize; x++) {
                   for (int y=0; y < c.size; y++) {
                       dW_Out[x][y] += dlogits[x] * prevH[y];
                   }
               }

               for (int y=0; y < c.size; y++) {
                   dH[y] = 0;
                   for (int x=0; x < c.inputSize; x++) {
                       dH[y] += c.W_Out[x][y] * dlogits[x];
                   }
               }

               for (int j=0; j<c.size; j++) {
                   //step 2
                   d0[j]   = dH[j] * tanh(C_t[j]);
                   dC_t[j] = dH[j] * o[j] * (1 - pow(tanh(C_t[j]), 2));

                   dF[j] = dC_t[j] * prevC[j];
                   dPrevC[j] = dC_t[j] * f[j];

                   dIt[j] = dC_t[j] * g[j];
                   dG[j] = dC_t[j] * it[j];

                   dZ_f[j] = dF[j] * f[j] * (1 - f[j]);
                   dZ_it[j] = dIt[j] * it[j] * (1 - it[j]);
                   dZ_g[j] = dG[j] * g[j] * (1 - pow(g[j], 2));
                   dZ_o[j] = d0[j] * o[j] * (1 - o[j]);
               }

               for (int j=0; j < c.Forget.size(); j++) {
                   for (int w = 0; w < c.size; w++) {
                       dW_Forget[j][w] += dZ_f[j] * input[w];
                   }
                   dBias_Forget[j] += dZ_f[j];
               }

               for (int j=0; j < c.State.size(); j++) {
                   for (int w = 0; w < c.size; w++) {
                       dW_State[j][w] += dZ_g[j] * input[w];
                   }
                   dBias_State[j] += dZ_g[j];
               }

               for (int j=0; j < c.Output.size(); j++) {
                   for (int w = 0; w < c.size; w++) {
                       dW_Output[j][w] += dZ_o[j] * input[w];
                   }
                   dBias_Output[j] += dZ_o[j];
               }

               for (int j=0; j < c.Input.size(); j++) {
                   for (int w = 0; w < c.size; w++) {
                       dW_Input[j][w] += dZ_it[j] * input[w];
                   }
                   dBias_Input[j] += dZ_it[j];
               }

               for (int j=0; j < c.size; j++) {
                   for (int w = 0; w < input.size(); w++) {
                       c.Forget[j].weights[w].value -= learningRate * dW_Forget[j][w];
                       c.Input[j].weights[w].value -= learningRate * dW_Input[j][w];
                       c.Output[j].weights[w].value -= learningRate * dW_Output[j][w];
                       c.State[j].weights[w].value -= learningRate * dW_State[j][w];
                   }
                   c.Forget[j].bias -= learningRate * dBias_Forget[j];
                   c.Output[j].bias -= learningRate * dBias_Output[j];
                   c.Input[j].bias -= learningRate * dBias_Input[j];
                   c.State[j].bias -= learningRate * dBias_State[j];
               }

               for (int y=0; y < c.size; y++) {
                   for (int x=0; x < c.inputSize; x++) {
                       c.W_Out[x][y] -= learningRate * dW_Out[x][y];
                   }
               }
           }

           //store states
           c.h_hist.push_back(prevH);
           c.c_hist.push_back(prevC);
       }

        if (e % 1000 == 0) {
            cout << "Total loss: " << totalLoss << endl;
        }
    }
}

int LSTM::activate(vector<int> input) {
    vector<double> prevH = c.h_hist.empty() ? vector<double>(c.size, 0.0) : c.h_hist.back();
    vector<double> prevC = c.c_hist.empty() ? vector<double>(c.size, 0.0) : c.c_hist.back();

    vector<double> C_t(c.size, 0.0);
    vector<double> H_t(c.size, 0.0);

    for (int i=0; i < input.size(); i++) {
        int curToken = input[i];

        vector<double> inp;
        inp.push_back((double)curToken);
        for (int h = 0; h < prevH.size(); h++) {
            inp.push_back(prevH[h]);
        }

        vector<double> f(c.size, 0.0), it(c.size, 0.0), g(c.size, 0.0), o(c.size, 0.0);

        for (int i = 0; i < c.Forget.size(); i++) f[i]  = c.Forget[i].activate(inp);
        for (int i = 0; i < c.Input.size();  i++) it[i] = c.Input[i].activate(inp);
        for (int i = 0; i < c.State.size();  i++) g[i]  = c.State[i].activate(inp);
        for (int i = 0; i < c.Output.size(); i++) o[i]  = c.Output[i].activate(inp);

        for (int j = 0; j < c.size; j++) {
            C_t[j] = (f[j] * prevC[j]) + (it[j] * g[j]);
            H_t[j] = o[j] * tanh(C_t[j]);
        }

        c.h_hist.push_back(H_t);
        c.c_hist.push_back(C_t);

        prevH = H_t;
        prevC = C_t;
    }

    vector<double> logits(c.inputSize, 0.0);
    for (int x = 0; x < c.inputSize; x++) {
        for (int y = 0; y < c.size; y++) {
            logits[x] += c.W_Out[x][y] * H_t[y];
        }
    }

    double maxLogit = *max_element(logits.begin(), logits.end());
    double sumExp = 0.0;
    for (int v = 0; v < logits.size(); v++) sumExp += exp(logits[v] - maxLogit);

    vector<double> probs(c.inputSize);
    for (int v = 0; v < logits.size(); v++) probs[v] = exp(logits[v] - maxLogit) / sumExp;

    int predicted = max_element(probs.begin(), probs.end()) - probs.begin();
    return predicted;
}