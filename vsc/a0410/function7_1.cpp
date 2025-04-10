#include <stdio.h>

void myFunction() {
    int x = 5;
}

int main() {
    myFunction();
    // printf("%d", x); 지역변수 접근 오류

    return 0;
}