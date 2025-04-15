#include <iostream>
#include <list>
#include <string>
using namespace std;

int main() {
    list<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};
    cars.pop_front();
    cars.pop_back();
    for(string car : cars) {
        cout << car << endl;
    }

    return 0;
}
