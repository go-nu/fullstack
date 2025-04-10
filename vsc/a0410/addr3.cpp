#include <stdio.h>

int main() {
    int myAge = 43;

    int *ptr = &myAge;

    printf("%d\n", myAge); // myAge 값
    printf("%p\n", &myAge); // myAge 주소
    printf("%p\n", ptr); // 포인터 ptr의 주소
    

    return 0;
}