#include <stdio.h>

int main() {
    int myNumbers[] = {10, 25, 50, 75, 100};
    int length = sizeof(myNumbers) / sizeof(myNumbers[0]);
    printf("%d ", length);

    return 0;
}
// int 는 보통 4 byte를 사용, 4*5 = 20