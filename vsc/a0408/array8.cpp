#include <stdio.h>

int main() {
    int ages[] = {20, 22, 18, 35, 48, 26, 87, 70};
    int min = ages[0];
    int i, j;
    int length = sizeof(ages) / sizeof(ages[0]);
    for (i = 0; i < length; i++) {
        if(min > ages[i]) {
            min = ages[i];
        }
    }
    
    printf("최소나이 %d", min);

    return 0;
}