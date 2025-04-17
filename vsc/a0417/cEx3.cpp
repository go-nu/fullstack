#include <stdio.h>
#include <string.h>

int main() {
    char str[100];
    printf("문자열 입력: ");
    scanf("%s", str);
    int size = strlen(str);

    for(int i = 0; i < size/2; i++) {
        char temp = str[i];
        str[i] = str[size-i-1] ; // str 맨 뒤에 \n이 있어서 -1 해줘야함.
        str[size-i-1] = temp;
    }
    printf("%s", str);

    return 0;
}