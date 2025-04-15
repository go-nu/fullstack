#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};
    cars.pop_back(); // 마지막 요소 제거

    for(string car : cars) {
        cout << car << "\n";
    }
    cout << cars.size() << endl;
    cout << cars.empty() << endl; // 비어있으면 1 있으면 0
    return 0;
}
