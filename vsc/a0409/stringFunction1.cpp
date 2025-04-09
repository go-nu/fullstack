#include <stdio.h>
#include <string.h>

int main() {
    char alphabet[50] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    printf("%d", strlen(alphabet)); // 26
    printf(" %d", sizeof(alphabet)); // 50

    return 0;
}