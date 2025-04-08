#include <stdio.h>

int main() {
    
    for (int i = 1; i <= 2; i++) {
        printf("Outer : %d\n", i);
        for (int j = 1; j <=3; j++) {
            printf("Inner : %d\n", j);
        }
    }
    

    return 0;
}