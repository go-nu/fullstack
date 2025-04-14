#include <iostream>
#include <string>
using namespace std;

class MyClass {
    public:
        int x;
    
    private: 
        int y;

};

int main() {
    MyClass myObj;
    myObj.x = 25;
    // myObj.y = 50; // private 접근 불가

    return 0;
}