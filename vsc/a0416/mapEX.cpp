#include <iostream>
#include <map>
using namespace std;

int main() {
    map<string, int> score;

    score["Lee"] = 5200;
    score["Kim"] = 4500;
    score["Park"] = 4800;
    score["Choi"] = 5500;
    score["Jung"] = 6100;

    cout << "연봉이 5000만원 이상인 직원들: " << endl;
    for(auto it = score.begin(); it != score.end(); ++it) {
        if((it -> second) >= 5000) {
            cout << it -> first << ": " << it -> second << endl;
        }
    }

    return 0;
}