#include<stdio.h>
int main(){
    int a, b, c;
    printf("숫자 3개 입력 : ");
    scanf("%d %d %d", &a, &b, &c);
   
    int max = (a > b) ? a : b;
    max = (max > c) ? max : c;
    
    printf("가장 큰 수는 %d입니다.\n",max);
    return 0;
}