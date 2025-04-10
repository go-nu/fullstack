#include <stdio.h>

int main() {
    int myNumbers[4] = {25, 50, 75, 100};
    int i;
    
    for (i = 0; i < 4; i++) {
      printf("%p\n", &myNumbers[i]);
    }

    return 0;
}
// int = 4byte, 메모리 주소가 4씩 차이남