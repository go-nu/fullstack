#include <stdio.h>

int main() {
    int myNum;
    printf("Type a number and press enter: \n");
    scanf("%d", &myNum); // & 주소를 통해 값을 저장
    printf("Your number is : %d", myNum);

    return 0;
}
// %d 정수 - int
// %f 실수 - float
// %lf 실수 - double
// %c 문자 - char
// %s 문자열 - char[]