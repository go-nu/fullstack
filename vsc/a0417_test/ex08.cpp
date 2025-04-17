#include <iostream>
#include <string>
using namespace std;

class Student {
    public:
        string name;
        int id;
        Student(string n, int i) {
            name = n;
            id = i;
            cout << name << "[" << id << "] 객체를 생성했습니다.\n";
        }
        ~Student() {
            cout << name << "객체가 삭제되었습니다.\n";
        }
};

int main() {
    Student s1("kim", 1);
    Student s2("lee", 2);
}