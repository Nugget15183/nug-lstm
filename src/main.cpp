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

    lstm.train(trainingData, trainingDataTargets, 10000);

    int times = 10;
    char last = *"c";

    //lstm.reset();
    lstm.activate(trainingData[0]);

    for (int i=0; i<times; i++) {
        int token = last - 'a' + 1;
        int result = lstm.activate({token});

        char lowerLetter = 'a' + (result - 1);
        last = lowerLetter;

        cout << lowerLetter << endl;
    }

    return 0;
}