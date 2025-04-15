#include <iostream>
#include <string>
using namespace std;

class Member {
    public:
        string name;
        int age;
        string address;
        Member(){};
        Member(string n, int a, string ad) {
            name = n;
            age = a;
            address = ad;
        }
};

int main() {
    Member m1;
    m1.name = "kim";
    m1.age = 25;
    m1.address = "seoul";
    Member m2 = Member("park", 26, "Busan");

    cout << m1.name << "은 " << m1.age << "살이고, " << m1.address <<"에 거주한다.\n";
    cout << m2.name << "은 " << m2.age << "살이고, " << m2.address <<"에 거주한다.";

    return 0;
}