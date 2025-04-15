#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<int> numbers;
    int n;
    cout << "정수 5개 입력: ";
    for(int i = 0; i < 5; i++) {
        cin >> n;
        numbers.push_back(n);
    }

    for(int number : numbers) {
        cout << number << endl;
    }
    return 0;
}
