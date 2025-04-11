#include <iostream>
#include <string>
using namespace std;

int main() {
    struct {
        int myNum;
        string myStrig;
    }myStruct;

    myStruct.myNum = 1;
    myStruct.myStrig = "Hello";

    cout << myStruct.myNum << "\n";
    cout << myStruct.myStrig << "\n";
    
    return 0;
}