#include <iostream>
#include <string>
using namespace std;

class MyClass {
    public:
        void myMethod();

};

// 외부에서 클래스 내부의 함수를 정의하려면, 클래스 내부에서 함수를 선언한 뒤 클래스 이부에서 정의.
// 클래스 이름, 범위 결정 :: 연산자, 함수이름   을 차례로 지정.
void MyClass::myMethod() {
    cout << "Hello World!";
}

int main() {
    MyClass myObj;
    myObj.myMethod();    

    return 0;
}