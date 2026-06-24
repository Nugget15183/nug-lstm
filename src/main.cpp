#include <iostream>
#include <vector>
#include <fstream>
#include "util.h"
#include "LSTM.h"

using namespace std;

int main() {
    LSTM lstm = LSTM();

    string test = "abcabcabc";
    string test_target = test.substr(1) + test.substr(0, 1);

    vector<vector<int>> trainingData = {tokenize(test)};
    vector<vector<int>> trainingDataTargets = {tokenize(test_target)};

    lstm.train(trainingData, trainingDataTargets, 1000);

    return 0;
}