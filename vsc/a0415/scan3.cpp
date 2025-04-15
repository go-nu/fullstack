#include <stdio.h>

int main() {
    int a, b, c;
    printf("숫자 3개 입력 : ");
    scanf("%d %d %d", &a, &b, &c);
    (a>=b) ? ((a >= c)? printf("가장 큰 수는 %d입니다.", a) : printf("가장 큰 수는 %d입니다.", c)) : ((b >= c)? printf("가장 큰 수는 %d입니다.", b) : printf("가장 큰 수는 %d입니다.", c));
    // if(a >= b) {
    //     if (a >= c) {
    //         printf("가장 큰 수는 %d입니다.", a);
    //     } else printf("가장 큰 수는 %d입니다.", c);
    // } else {
    //     if (b >= c) {
    //         printf("가장 큰 수는 %d입니다.", b);
    //     } else printf("가장 큰 수는 %d입니다.", c);
    // }
}