#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};
    // cars[i] = cars.at(i)
    // 대신 오류에 대해 자세히 알려줌.
    cout << cars.at(1) << "\n";
    cout << cars.at(2) << "\n";
    cout << cars.at(6) << "\n";

    return 0;
}
