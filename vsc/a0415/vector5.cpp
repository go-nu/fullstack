#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};
    cars.push_back("렉서스"); // 마지막에 추가하기
/*
    cars[0] = "테슬라";
    cars.at(1) = "Hyundai";
    cout << cars[0] << "\n";
    cout << cars.at(1) << "\n";
*/
    for(string car : cars) {
        cout << car << "\n";
    }

    return 0;
}
