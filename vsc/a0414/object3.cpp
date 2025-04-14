#include <iostream>
#include <string>
using namespace std;

class MyClass {
    public:
        void myMethod() {
            cout << "Hello World!";
        }

};

int main() {
    MyClass myObj;
    myObj.myMethod();    

    return 0;
}