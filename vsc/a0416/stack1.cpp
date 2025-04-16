#include <iostream>
#include <stack>
using namespace std;

int main() {
    stack<int> s; // LIFO
    // 스택에 값 추가
    s.push(10);
    s.push(20);
    s.push(30);
    cout << "현재 스택의 맨 위 값: " << s.top() << endl;

    s.pop();
    cout << "현재 스택의 맨 위 값: " << s.top() << endl;

    if(!s.empty()) {
        cout << "스택은 비어있지 않습니다." << endl;
    }

    cout << "현재 스택의 크기: " << s.size() << endl;

    while(!s.empty()) {
        cout << "스택에서 꺼낸 값: " << s.top() << endl;
        s.pop();
    }

    return 0;
}
// push() 값 넣기
// pop() 값 빼기
// top() 맨 위 값 확인
// empty() 비어있는가
// size() 크기