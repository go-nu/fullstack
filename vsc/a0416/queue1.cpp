#include <iostream>
#include <queue>
using namespace std;

int main() {
    queue<int> q; // FIFO
    
    q.push(10);
    q.push(20);
    q.push(30);
    cout << "현재 큐의 맨 앞 값: " << q.front() << endl;
    cout << "현재 큐의 맨 뒤 값: " << q.back() << endl;

    q.pop();
    cout << "pop()이후 맨 앞의 값: " << q.front() << endl;

    cout << "현재 큐의 크기: " << q.size() << endl;

    while(!q.empty()) {
        cout << "큐에서 꺼낸 값: " << q.front() << endl;
        q.pop();
    }


    return 0;
}
// push() 값 넣기 (뒤로)
// pop() 값 빼기 (앞으로)
// front() 맨 앞 값 확인
// back() 맨 뒤 값 확인
// empty() 비어있는가
// size() 크기