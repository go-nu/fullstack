#include <iostream>
#include <deque>
using namespace std;

int main() {
    deque<int> dq; // FIFO
    
    dq.push_back(10);
    dq.push_back(20);
    dq.push_front(5);
    cout << "맨 앞 값: " << dq.front() << endl;
    cout << "맨 뒤 값: " << dq.back() << endl;

    dq.pop_front();
    dq.pop_back();

    dq.push_back(30);
    dq.push_front(1);

    cout << "현재 데큐 요소들" << endl;
    for(int num : dq) {
        cout << num << " ";
    }

    return 0;
}
// push_back() 뒤에 값 넣기
// push_front() 앞에 값 넣기
// pop_back() 뒤에 값 빼기
// pop_front() 앞에 값 빼기
// front() 맨 앞 값 확인
// back() 맨 뒤 값 확인
// size() 크기
// empty() 비어있는가