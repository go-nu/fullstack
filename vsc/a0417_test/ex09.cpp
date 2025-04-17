#include <iostream>
#include <string>
using namespace std;

class Animal{
    public:
        void sound() {
            cout << "animal make sound\n";
        }
};

class Dog : public Animal{
    public : 
        void sound() {
            cout << "멍멍!\n";
        }
};

int main() {
    Dog d;
    d.sound();

    return 0;
}