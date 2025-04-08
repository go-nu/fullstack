#include <stdio.h>

int main() {
    int myNumbers[] = {10, 25, 50, 75, 100};
    
    printf("%lu", sizeof(myNumbers)); // unsigned long;

    return 0;
}
// int 는 보통 4 byte를 사용, 4*5 = 20