#include <stdio.h>

myFunction(int x, int y) {
    return y + x;
}

int main() {
    printf("결과 값은 : %d", myFunction(5, 3));

    return 0;
}