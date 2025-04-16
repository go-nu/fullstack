#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> mySet;

    // 추가 (중복 불가)
    mySet.insert(50);
    mySet.insert(20);
    mySet.insert(30);
    mySet.insert(20); // 중복 값은 무시

    for(int num : mySet) {
        cout << num << " ";
    }
    cout << "\n";

    // 값 찾기
    if(mySet.find(30) != mySet.end()) {
        cout << "30을 찾았습니다." << endl;
    } else {
        cout << "30은 없습니다." << endl;
    }

    // 값 삭제
    mySet.erase(20);
    cout << "20 삭제 후 set: ";
    for(int num : mySet) {
        cout << num << " ";
    }

    return 0;
}
// insert() 원소 삽입(중복x)
// find() 찾기 (반환값 == end()이면 없음)
// erase() 특정 원소 제거
// size() 크기
// empty() 비어있는지
// clear() 원소 모두 제거