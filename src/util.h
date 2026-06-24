#ifndef LSTM_UTIL_H
#define LSTM_UTIL_H

#include <cctype>
#include <iosfwd>
#include <string>
#include <vector>
#include <sstream>
#include <bitset>
#include <fstream>

using namespace std;

class util {
public:
    string vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'\"-;:\n";

    vector<int> tokenize(string text) {
        vector<int> result;
        result.reserve(text.size());

        for (int i=0; i<text.size(); i++) {
            char letter = text[i];
            int index = vocab.find(letter);
            if (index != string::npos) {
                result.push_back(index);
            }
        }

        return result;
    }

    string read_txt(string filename) {
        ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error opening file!" << std::endl;
            return "";
        }

        ostringstream ss;
        ss << file.rdbuf();

        string fileContents = ss.str();
        cout << fileContents << endl;
        return fileContents;
    }

};

#endif //LSTM_UTIL_H
