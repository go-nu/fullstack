#include <stdio.h>
#include <string.h>

int main() {
    char str1[] = "Hello";
    char str2[] = "Hello";
    char str3[] = "Hi";
    // string compare, 같으면 0, 다르면 음수 출력
    printf("%d\n", strcmp(str1, str2));
    printf("%d\n", strcmp(str1, str3));

    return 0;
}