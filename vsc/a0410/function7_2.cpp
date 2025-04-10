#include <stdio.h>

int x = 5; // global variation 전역 변수
void myFunction() {
    printf("%d\n", x);

}

int main() {
    myFunction();
    printf("%d\n", x);

    return 0;
}