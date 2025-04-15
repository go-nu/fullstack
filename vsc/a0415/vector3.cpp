#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};

    cout << cars.front() << "\n"; // 첫번째 요소 접근
    cout << cars.back() << "\n"; // 마지막 요소 접근

    return 0;
}
