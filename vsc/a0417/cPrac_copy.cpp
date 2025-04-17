#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> v;
    int vs;

    cout << "벡터 크기 입력: ";
    cin >> vs;

    int num;
    cout << "정수 입력:\n";
    for(int i = 0; i < vs; i++) {
        cin >> num;
        v.push_back(num);
    }

    int sum = 0;
    for(int i = 0; i < vs; i++) {
        sum += v[i];
    }

    cout << "합: " << sum << endl;
    cout << "벡터: ";
    for(int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << endl;

    return 0;
}