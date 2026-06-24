#include <iostream>
#include <vector>
#include <fstream>
#include "util.h"

using namespace std;

int main() {
    string test = "The quick brown fox jumps over the lazy dog";

    vector<int> tokens = tokenize(test);

    for (int i=0; i<tokens.size(); i++) {
        int token = tokens[i];

        cout << token << endl;
    }

    return 0;
}