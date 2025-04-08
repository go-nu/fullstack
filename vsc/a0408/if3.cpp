#include <stdio.h>

int main() {
    int myNum = 5;
    if ((myNum % 2) == 0) {
        printf("%d는 짝수입니다.", myNum);
    } else {
        printf("%d는 홀수입니다.", myNum);
    }
    
    
    return 0;
}