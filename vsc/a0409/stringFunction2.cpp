#include <stdio.h>
#include <string.h>

int main() {
    char str1[20] = "Hello ";
    char str2[] = "World!";
    // string concate, str1에 str2 이어 붙이기
    strcat(str1, str2);

    printf("%s\n", str1);

    return 0;
}