#include <stdio.h>

int main() {
    int num;
    int sum = 0;
    printf("양의 정수를 입력하세오. : ");
    scanf("%d", &num);

    if(num < 0) {
        printf("양의 정수를 입력하세요");
        return 1;
    }
    while (num > 0) {
        sum = sum + (num%10);
        num = num / 10;
    }
    printf("각 자릿수의 합: %d", sum);

    return 0;
}