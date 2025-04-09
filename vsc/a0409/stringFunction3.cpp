#include <stdio.h>
#include <string.h>

int main() {
    char str1[20] = "Hello World!";
    char str2[20];
    // string copy, str2에 str1 복사
    strcpy(str2, str1);

    printf("%s\n", str1);
    printf("%s\n", str2);

    return 0;
}