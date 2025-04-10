#include <stdio.h>

void calculaeSum(int n1, int n2) {
    int sum = n1 + n2;
    printf("%d + %d = %d\n", n1, n2, sum);
}

int main() {
    calculaeSum(5, 3);
    calculaeSum(8, 2);
    calculaeSum(15, 15);

    return 0;
}