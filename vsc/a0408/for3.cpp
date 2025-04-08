#include <stdio.h>

int main() {
    // 2의 거듭제곱에서 512까지의 거듭제곱을 출력
    for (int i = 2; i <= 512; i*=2) {
        printf("%d ", i);
    }
    

    return 0;
}