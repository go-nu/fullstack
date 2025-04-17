#include <iostream>
#include <string>
using namespace std;
#include <vector>

int main() {
    vector<int> v;
    int num;

    while (true) {
        cout << "\n정수 입력(-1 종료) : ";
        cin >> num;
        if (num == -1) {
            break;
        }
        else {
            v.push_back(num);
        }
        cout << "벡터의 모든 값 : ";
        for(int i = 0; i < v.size(); i ++) {
            cout << v[i] << " ";
        }
    }

}