#include <stdio.h>

int main() {
    int arr[5];
    printf("정수 5개 입력 : ");
    for(int i = 0; i < 5; i++) {
        scanf("%d", &arr[i]);
    }
    int max = arr[0];
    for(int i = 0; i < 5; i++) {
        if(arr[i] > max) {
            max = arr[i];
        }
    }
    printf("가장 큰 값 : %d", max);
}