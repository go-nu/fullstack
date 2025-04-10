#include <stdio.h>

int x = 5; // global variation 전역 변수
void myFunction() {
    int x = 22; // local varitaion 지역 변수
    printf("%d\n", x);

}

int main() {
    myFunction();
    printf("%d\n", x);

    return 0;
}