#include <iostream>
#include <string>
using namespace std;

int main() {
    string name;
    int age;

    cout << "사용자 이름 입력 : ";
    cin >> name;
    cout << "사용자 나이 입력 : ";
    cin >> age;

    cout << name << "님은 " << age << "세입니다.\n";

    return 0;
}