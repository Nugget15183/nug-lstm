#ifndef LSTM_UTIL_H
#define LSTM_UTIL_H

#include <cctype>
#include <iosfwd>
#include <string>
#include <vector>
#include <sstream>
#include <bitset>

using namespace std;

vector<int> tokenize(string text) {
    istringstream iss(text);
    vector<string> words;
    string word;

    while (iss >> word) {
        words.push_back(word);//split input string into smaller strings that are each word
    }

    vector<int> result;
    result.reserve(words.size());

    for (int i = 0; i < words.size(); i++) {
        string curword = words[i];

        for (char c : curword) {
            if (std::isalpha(c)) {
                int number = std::toupper(c) - 'A' + 1;
                result.push_back(number);//convert each letter of each word into its number value (e.g a = 1, b = 2)
            }
        }
    }

    return result;
}

#endif //LSTM_UTIL_H
