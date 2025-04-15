#include <stdio.h>

int main() {
    int numbers[5];
    int i;
    int temp = 0;
    printf("숫자 5개 입력 : ");
    // 배열 생성
    for (i = 0; i < 5; i ++) {
        scanf("%d", &numbers[i]);
    }
    // 배열에서 값 비교하며 가장 큰 수 찾기
    for (i = 0; i < 5; i ++) {
        if (numbers[i] > temp) {
            temp = numbers[i];
        }        
    }
    printf("가장 큰 숫자는 %d이입니다.", temp);

}