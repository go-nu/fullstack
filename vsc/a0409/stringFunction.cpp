#include <stdio.h>
#include <string.h>

int main() {
    char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    // str의 길이를 int형으로 반환, **include <string.h>**
    printf("%d", strlen(alphabet)); // 26
    // sizeof는 null문자 포함
    printf("%d", sizeof(alphabet)); // 27

    return 0;
}