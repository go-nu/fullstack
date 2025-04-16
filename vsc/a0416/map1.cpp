#include <iostream>
#include <map>
using namespace std;

int main() {
    map<string, int> score;

    score["Alice"] = 95;
    score["Bob"] = 82;
    score["Charlie"] = 88;

    cout << "학생 점수 목록: " << endl;
    // for(map<string, int>::iterator it = score.begin(); it != score.end(); ++it) {
    //     cout << it -> first << ": " << it -> second << endl;
    // }
    for(auto it = score.begin(); it != score.end(); ++it) {
        cout << it -> first << ": " << it -> second << endl;
    }
    // it -> first : key
    // it -> second : value

    string name = "Bob";
    if(score.find(name) != score.end()) {
        cout << name << "의 점수는 " << score[name] << "점 입니다." << endl;
    } else {
        cout << name << "은 존재하지 않습니다." << endl;
    }

    // 삭제
    score.erase("Charlie");
    cout << "삭제 후 남은 학생 목록" << endl;
    for(const auto& pair : score) {
        cout << pair.first << ": " << pair.second << endl;

    }
    // score라는 map의 각 K-V 쌍을 pair라는 이름으로 하나씩 가져와 score 대상으로 순회한다.
    // pair.first - 현재 순회중인 key
    // pair.second - 현재 순회중인 value

    return 0;
}
// insert() 원소 삽입(중복x)
// find() 찾기 (반환값 == end()이면 없음)
// erase() 특정 원소 제거
// size() 크기
// empty() 비어있는지
// clear() 원소 모두 제거