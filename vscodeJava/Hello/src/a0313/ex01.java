package a0313;

public class ex01 {
    public static void main(String[] args) {
        // 세자리 정수의 각 자릿수 총 합을 출력하려고 한다.
        // num = 374
        // 정수 374의 각 자릿수의 총합 : 14
        int num = 374;
        int a = num / 100;
        int b = (num % 100) / 10; // (num / 10 % 10)
        int c = num % 10;
        int sum = a + b + c;
        System.out.printf("정수 %d의 각 자릿수의 총합 : %d", num, sum);
    }
}
