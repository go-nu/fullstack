#include <iostream>
#include <string>
using namespace std;

class Animal{
    public:
        string name;
        void say() {
            cout << "make sound\n";
        }
        ~Animal(){
            cout<<"Animal close\n";
        }
};
class Dog : public Animal {
    public: 
        Dog(){
            name = "Dog";
        }
        ~Dog(){
            cout << "Dog close\n";
        }
        void say(){
            cout << "bark\n";
        }
};
class Cat : public Animal {
    public:
        Cat(){
            name = "Cat";
        }
        ~Cat(){
            cout << "Cat close\n";
        }
        void say(){
            cout << "meow\n";
        }
};

int main() {
    Animal a;
    Dog d;
    Cat c;
    a.say();
    d.say();
    c.say();

    return 0;
}