#include <iostream>
#include <string>
using namespace std;

class Car {
    public:
        string modelName;
        int speed;
        Car(string m, int s) {
            modelName = m;
            speed = s;
        }
        int speedUp() {
            speed += 10;
            cout << modelName << ", now speed: " << speed << "\n";
            return speed;
        }
        int speedDown() {
            speed -= 10;
            cout << modelName << ", now speed: " << speed << "\n";
            return speed;
        }
};

int main() {
    Car car("bongbong", 60);
    car.speedUp();
    car.speedDown();
    car.speedDown();
    
}