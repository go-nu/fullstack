#include <stdio.h>

int main() {
    // Create a string
    char firstName[30];

    // Ask the user to input some text (name)
    printf("Enter your first name and press enter: \n");

    // Get and save the text
    scanf("%s", firstName);
    // string은 &를 사용하지 않음?
    // 배열 자체가 주소이므로 &를 통해 주소에 접근하지 않아도 된다.

    // Output the text
    printf("Hello %s", firstName);

    return 0;
}
// %d 정수 - int
// %f 실수 - float
// %lf 실수 - double
// %c 문자 - char
// %s 문자열 - char[]