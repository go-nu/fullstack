package a0317;

import java.util.Scanner;

public class Calculate {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        boolean flag = true;

        int n1 = 0;
        int n2 = 0;
        int result = 0;

        while (flag) {
            System.out.println("------------------------------------------------");
            System.out.println("1. 덧셈 | 2. 뺄셈 | 3. 곱셈 | 4. 나눗셈 | 5. 종료");
            System.out.println("------------------------------------------------");
            System.out.print("선택>");
            int num = s.nextInt();
            if (num == 5) {
                flag = false;
                continue;
            }
            System.out.print("첫 번째 숫자>");
            n1 = s.nextInt();
            System.out.print("두 번째 숫자>");
            n2 = s.nextInt();
            switch (num) {
                case 1:
                    result = n1 + n2;
                    System.out.printf("결과 : %d + %d = %d \n", n1, n2, result);
                    break;
                case 2:
                    result = n1 - n2;
                    System.out.printf("결과 : %d - %d = %d \n", n1, n2, result);
                    break;
                case 3:
                    result = n1 * n2;
                    System.out.printf("결과 : %d * %d = %d \n", n1, n2, result);
                    break;
                case 4:
                    if (n2 == 0) {
                        System.out.println("0으로 나눌 수 없습니다.");
                    } else {
                        result = n1 / n2;
                        System.out.printf("결과 : %d / %d = %d \n", n1, n2, result);
                    }
                    break;
                default:
                    break;
            }
            System.out.println();
        }
        System.out.println("프로그램 종료");
        s.close();
    }
}
