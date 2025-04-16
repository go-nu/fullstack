#include <iostream>
#include <string>
using namespace std;
#include <vector>

class Book {
    public:
        string title;
        bool state;

        Book(string t) {
            title = t;
            state = true;
        }
};

string bookState(bool b) {
    if(b) return "대출가능";
    else return "대출중";
}

int main() {
    vector<Book> books;
    books.push_back(Book("C++ 입문서"));
    books.push_back(Book("자료구조론"));
    books.push_back(Book("알고리즘 기초"));
    int choice;

    while(true) {
        cout << "\n=== 도서 대출 프로그램 ===" << endl;
        cout << "1. 도서 목록 확인" << endl;
        cout << "2. 도서 대출" << endl;
        cout << "3. 도서 반납" << endl;
        cout << "4. 프로그램 종료" << endl;
        cout << "메뉴 선택: ";
        cin >> choice;

        if (choice == 1) {
            cout << "[도서 목록]" << endl;
            for (int i = 0; i < books.size(); i++) {
                cout << (i+1) << ". " << books[i].title << "(" << bookState(books[i].state) << ")\n";
            }
        } else if (choice == 2) {
            int c1;
            cout << "대출할 도서 번호를 입력하세요: ";
            cin >> c1;
            if(c1 < 1 || c1 > books.size()) {
                cout << "잘못된 도서 번호입니다.\n"
            }
            else if (books[c1-1].state) {
                cout << books[c1-1].title << "책을 대출했습니다.\n";
                books[c1-1].state = false;
            } else cout << "이미 대출중인 책입니다.\n";
        } else if (choice == 3) {
            int c2;
            cout << "반납할 도서 번호를 입력하세요: ";
            cin >> c2;
            if(c2 < 1 || c2 > books.size()) {
                cout << "잘못된 도서 번호입니다.\n"
            }
            else if (!books[c2-1].state) {
                cout << books[c2-1].title << "책을 반납했습니다.\n";
                books[c2-1].state = true;
            } else cout << "이미 반납된 책입니다.\n";
            
        } else if (choice == 4) {
            cout << "프로그램을 종료합니다\n";
            break;
        } else cout << "잘못된 선택입니다. 다시 입력해주세요.";
    }

    return 0;
}
