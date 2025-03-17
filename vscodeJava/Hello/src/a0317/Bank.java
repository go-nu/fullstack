package a0317;

import java.util.Scanner;

public class Bank {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        boolean flag = true;

        int n1 = 0;
        int n2 = 0;
        int n3 = 0;

        while (flag) {
            System.out.println("----------------------------------------");
            System.out.println("1. 예금 | 2. 출금 | 3. 잔고 | 4. 종료");
            System.out.println("----------------------------------------");
            System.out.print("선택>");
            int num = s.nextInt();
            switch (num) {
                case 1:
                    System.out.print("예금액>");
                    n1 = s.nextInt();
                    break;
                case 2:
                    System.out.print("출금>");
                    n2 = s.nextInt();
                    n3 = n1 - n2;
                    break;
                case 3:
                    System.out.print("잔고>" + n3);
                    System.out.println();
                    break;
                case 4:
                    System.out.println();
                    System.out.println("프로그램 종료");
                    flag = false;
                    break;
                default:
                    break;
            }

        }
        s.close();
    }
    
}
