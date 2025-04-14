#include <iostream>
#include <string>
using namespace std;

class myClass {
    public: 
        void myFunction() {
            cout << "Some Content in parent class";
        }
};

class myChild:public myClass {
};
class myGrandChild:public myChild {
};

int main() {
    myGrandChild myObj;
    myObj.myFunction();  
    

    return 0;
}