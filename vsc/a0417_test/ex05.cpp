#include <stdio.h>

int factorial(int n) {
    int result = 1;
    for(int i = n; i > 0; i--) {
        result *= i;
    }

    return result;
}

int main() {
    int num = 0;
    printf("정수 입력 : ");
    scanf("%d", &num);
    printf("%d 팩토리얼 = %d", num, factorial(num));
    return 0;
}