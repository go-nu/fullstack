#include <iostream>
#include <string>
using namespace std;

class Animal {
    public:
        void makeSound() {
            cout << "Animal makes sound\n";
        }
};

class Cat : public Animal {
    public:
        void makeSound() {
            cout << "cat says : meow\n";
        }
};
class Bird : public Animal {
    public:
        void makeSound() {
            cout << "bird says : jjaek\n";
        }
};

int main() {
    Animal a;
    Cat c;
    Bird b;

    a.makeSound();
    b.makeSound();
    c.makeSound();

    return 0;
}