#include <stdio.h>

void myFunction(); // 호출 하는 함수가 호출 보다 뒤에 위치하면, 함수선언을 먼저

// 절차지행적 (차례대로 읽어서 컴파일)
int main() {
    myFunction();

    return 0;
}

void myFunction() {
    printf("hello\n");
}
