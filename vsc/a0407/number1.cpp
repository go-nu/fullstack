#include <stdio.h>

int main()
{
    float myNum = 19.99;
    // printf("%f", myNum); // %1f는 소용 없음
    printf("%.1f", myNum);  // 소수점 아래 첫번째 자리(반올림)
    printf("%.2f\n", myNum);  // 전체넓이 10칸, 소수점 두번째 자리

    return 0;
}
