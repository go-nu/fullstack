package a0314;

import java.util.Scanner;

public class If3 {
    public static void main(String[] args) {
        // int price = 10000;
        // int age = 10;
        // scanner를 통해 가격과 나이를 입력받고, 총 할인금액과 결제금액 출력
        Scanner scan = new Scanner(System.in);
        System.out.print("결제 금액 입력 : ");
        int price = scan.nextInt();
        System.out.print("나이 입력 : ");
        int age = scan.nextInt();

        int discount = 0;

        if (price >= 10000) {
            discount += 1000;
            System.out.println("10000원 이상 구매, 1000원 할인");
        }
        if (age <= 10) {
            discount += 1000;
            System.out.println("어린이 1000원 할인");
        }
        System.out.println("총 할인 금액 : " + discount + "원, 총 결제 금액 : " + (price - discount) + "원");
        
        scan.close();
    }
}
