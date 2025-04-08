#include <stdio.h>

int main() {
    int i;
    int myNumbers[] = {25, 50, 75, 100};
    myNumbers[0] = 33;
    for(i = 0; i < 4; i++) {
        printf("%d ", myNumbers[i]);
    }

    return 0;
}