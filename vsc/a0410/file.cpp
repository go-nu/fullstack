#include <stdio.h>

int main() {
    FILE *fptr; // 기본적인 데이터 유형, 사용하려면 포인터 변수 fptr

    fptr = fopen("e:\\student\\filename.txt", "w");
    fclose(fptr);

    return 0;
}