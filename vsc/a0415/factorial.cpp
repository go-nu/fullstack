#include <stdio.h>

int factorial(int n) {
    int result = 1;
    for(int i = n; i > 0; i--) {
        result *= i;
    }
    return result;
}

int main() {
    int num;
    printf("양의 정수 입력: ");
    scanf("%d", &num);
    printf("%d! = %d", num, factorial(num));
    return 0;
}