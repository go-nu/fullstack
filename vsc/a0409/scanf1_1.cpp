#include <stdio.h>

int main() {
    int myNum;
    char myChar;
    printf("Type a number a character and press enter: \n");
    scanf("%d %c", &myNum, &myChar); // & 주소를 통해 값을 저장
    printf("Your number is : %d\n", myNum);
    printf("Your character is : %c\n", myChar);

    return 0;
}
// %d 정수 - int
// %f 실수 - float
// %lf 실수 - double
// %c 문자 - char
// %s 문자열 - char[]