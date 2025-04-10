#include <stdio.h>

cal (int x, int y) {
    return y + x;
}

int main() {
    int result1 = cal(5, 3);
    int result2 = cal(8, 2);
    int result3 = cal(15, 15);
    printf("결과 값은 : %d", result1);
    printf("결과 값은 : %d", result2);
    printf("결과 값은 : %d", result3);

    return 0;
}