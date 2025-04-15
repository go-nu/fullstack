#include <iostream>
#include <list>
#include <string>
using namespace std;

int main() {
    list<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};
    cars.push_front("Tesla");
    cars.push_back("V2");
    for(string car : cars) {
        cout << car << endl;
    }

    return 0;
}
