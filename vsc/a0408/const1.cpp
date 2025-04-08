#include <stdio.h>

int main() {
    const int myNum = 15;
    // myNum = 10; const 상수, 재할당 불가능


    // const int minutesPerHour;
    // minutesPerHour = 60; 
    // const는 선언과 동시에 초기화
    const int minutesPerHour = 60;

    printf("%d", myNum);
    
    return 0;
}