#include <iostream>
#include <string>
using namespace std;

int main() {
    string line;
    cout << "Type a full sentence: ";
    getline(cin, line);
    cout << "You wrote: " << line;

    return 0;
}
