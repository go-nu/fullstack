#include <iostream>
#include <string>
using namespace std;

class Car{
    public:
        int speed;
        string color;
};

int main() {
    Car c;
    c.speed = 60;
    c.color = "red";

    cout << c.color << "색 자동차의 속도는 " << c.speed <<"입니다.\n";
}