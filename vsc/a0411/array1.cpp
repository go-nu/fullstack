#include <iostream>
#include <string>
using namespace std;

int main() {
    /*
    string myString = "Hello";
    cout << myString[1] << "\n";
    cout << myString[myString.length()-1] << "\n";
    */
   int myNumbers[5] = {10, 20, 30, 40, 50};
    for (int i = 0; i < sizeof(myNumbers) / sizeof(myNumbers[0]); i++) {
    cout << myNumbers[i] << "\n";
    }
    return 0;
}