#include <stdio.h>
// #include <limits.h>

int main() {
    int i;
    int arr[10] = {5, 12, 8, 3, 15, 7, 9, 20, 1, 18};

    int sum =0;
    int length = sizeof(arr)/sizeof(arr[0]);
    for (i = 0; i < 10; i++) {
        sum += arr[i];
    }
    float result = (float) sum / length;
    printf("배열의 평균 : %.2f\n", result);

    int max = 0; //INT_MIN -2147483648(int형에서 제일 작은 수)
    for(i = 0; i < 10; i ++) {
        if (arr[i] >= max) {
            max = arr[i];
        }
    }
    printf("배열의 최댓값 : %d\n", max);

    int min = arr[0]; // INT_MAX 2147483647 (int형에서 제일 큰 수)
    for(i = 0; i < 10; i++) {
        if (min >= arr [i]) {
            min = arr[i];
        }
    }
    printf("배열의 최솟값 : %d\n", min);

    

    return 0;
}