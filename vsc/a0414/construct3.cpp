#include <iostream>
#include <string>
using namespace std;

class Car {
    public:
        string brand;
        string model;
        int year;
        Car(string x, string y, int z);
};
// 함수와 마찬가지로 생성자도 클래스 외부에서 정의할 수 있다.
// 클래수 내부에서 생성자를 선언한 뒤,  클래스 이름, 범위 :: 연산자, 생성자이름(클래스명과 동일) 을 차례로 지정.
Car::Car(string x, string y, int z) {
    brand = x;
    model = y;
    year = z;
}

int main() {
    Car carObj1("BMW", "X5", 1999);
    Car carObj2("Ford", "Mustang", 1969);
    cout << carObj1.brand << ", " << carObj1.model << ", " << carObj1.year << "\n";
    cout << carObj2.brand << ", " << carObj2.model << ", " << carObj2.year;

    return 0;
}