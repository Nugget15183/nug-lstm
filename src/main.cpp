#include <iostream>
#include <vector>
#include <fstream>
#include "util.h"
#include "LSTM.h"

using namespace std;

string datafile = "resources/save/state.bin";

int main() {
    util u = util();
    u.read_txt("resources/trainingdata/book1.txt");

    LSTM lstm = LSTM();
    bool loadedData = lstm.load(datafile);

    string test = "abcabcabc";
    string test_target = test.substr(1) + test.substr(0, 1);

    vector<vector<int>> trainingData = {u.tokenize(test)};
    vector<vector<int>> trainingDataTargets = {u.tokenize(test_target)};

    if (!loadedData) {
        lstm.train(trainingData, trainingDataTargets, 10000);
        lstm.save(datafile);
    }

    int times = 10;
    char last = *"c";

    lstm.reset();
    lstm.activate(trainingData[0]);

    for (int i=0; i<times; i++) {
        int token = u.vocab.find(last);
        int result = lstm.activate({token});

        last = u.vocab[result];

        cout << last << endl;
    }

    return 0;
}