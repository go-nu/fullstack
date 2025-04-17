#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};
    auto it = cars.begin() + 2; // .begin() + n => n번 index를 말함
    cars.insert(it, "hyndai");

    for (string car : cars) {
        cout << car << endl;
    }

    return 0;
}
