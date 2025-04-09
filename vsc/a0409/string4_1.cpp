#include <stdio.h>

int main() {
    char carName[] = "Volvo";
    int length = sizeof(carName) / sizeof(carName[0]); // 배열의 전체 byte / element 하나의 byte
    // char 하나 1byte 5글자 전체 5byte => 5/1 = 5
    int i;
    for (i = 0; i < 5; i++) {
        printf("%c", carName[i]);
    }

    return 0;
}