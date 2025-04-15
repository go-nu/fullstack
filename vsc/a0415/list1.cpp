#include <iostream>
#include <list>
#include <string>
using namespace std;

int main() {
    list<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};

    for(string car : cars) {
        cout << car << "\n";
    }

    return 0;
}
