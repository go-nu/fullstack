#include <iostream>
#include <string>
using namespace std;

class Book {
    public:
        string title;
        string author;
        int price;
        Book(string t, string a, int p) {
            title = t;
            author = a;
            price = p;
        }
        void printInfo(){
            cout << title << "[" << author << "]  " << price << "원\n";
        }
};

int main() {
    Book book("제목", "저자", 20000);
    book.printInfo();
}