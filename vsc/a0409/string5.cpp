#include <stdio.h>

int main() {
    char greetings[] = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!', '\0'};
    char greetings2[] = "Hello World!";

    // \0 : null문자
    // 12글자 + \0 포함해서 = 13글자
    printf("%lu\n", sizeof(greetings));
    printf("%lu", sizeof(greetings2));

    return 0;
}