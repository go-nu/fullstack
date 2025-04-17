#include <stdio.h>

int main() {
    int num;
    printf("정수 하나 입력 : ");
    scanf("%d", &num);

    if(num % 2 == 0) {
        printf("%d는 짝수입니다.", num);
    } else {
        printf("%d는 홀수입니다.", num);
    }

    return 0;
}