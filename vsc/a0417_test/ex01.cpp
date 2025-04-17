#include <stdio.h>

int main() {
    int n1, n2;
    printf("두 개의 정수 입력 : ");
    scanf("%d%d", &n1, &n2);

    int a1 = n1 + n2;
    int a2 = n1 - n2;
    int a3 = n1 * n2;
    int a4 = n1 / n2;
    int a5 = n1 % n2;

    printf("두 정수의 합 : %d\n", a1);
    printf("두 정수의 차 : %d\n", a2);
    printf("두 정수의 곱 : %d\n", a3);
    printf("두 정수의 몫 : %d\n", a4);
    printf("두 정수의 나머지 : %d\n", a5);
    return 0;
}