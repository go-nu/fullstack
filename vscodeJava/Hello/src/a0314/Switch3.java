package a0314;

import java.util.Scanner;

public class Switch3 {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        System.out.print("회원 등급 입력 : ");
        int grade = scan.nextInt();
        int coupon = switch (grade) {
            case 1 -> 1000;
            case 2 -> 2000;
            case 3 -> 3000;
            default -> 500;
        };
        System.out.println("발급받은 쿠폰 " + coupon);

        scan.close();
    }
}
