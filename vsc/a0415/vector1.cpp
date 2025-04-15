#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};

    for(string car : cars) {
        cout << car << "\n";
    }

    return 0;
}
