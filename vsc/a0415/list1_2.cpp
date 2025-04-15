#include <iostream>
#include <list>
#include <string>
using namespace std;

int main() {
    list<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};
    cars.front() = "Opel";
    cars.back() = "Toyota";
    cout << cars.front() << "\n"; 
    cout << cars.back() << "\n"; 

    return 0;
}
