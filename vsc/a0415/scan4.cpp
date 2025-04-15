#include <iostream>
using namespace std;

int main() {
    int a, b, c;
    printf("숫자 3개 입력 : ");
    cin >> a;
    cin >> b;
    cin >> c;
    
    if(a >= b) {
        if (a >= c) {
            cout << "가장 큰 수는 " << a << "입니다.";
        } else cout << "가장 큰 수는 " << c << "입니다.";
    } else {
        if (b >= c) {
            cout << "가장 큰 수는 " << b << "입니다.";
        } else cout << "가장 큰 수는 " << c << "입니다.";
    }
}