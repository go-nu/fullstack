#include <stdio.h>

int main() {
    int myNumbers[4] = {25, 50, 75, 100};
    int i;
    
    printf("%d\n", *myNumbers);
    printf("%d\n", *(myNumbers+1));
    printf("%d\n", *(myNumbers+2));
    
    return 0;
}
// int = 4byte, 메모리 주소가 4씩 차이남