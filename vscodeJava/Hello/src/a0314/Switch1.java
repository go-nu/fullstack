package a0314;

import java.util.Scanner;

public class Switch1 {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        System.out.print("회원 등급 입력 : ");
        int grade = scan.nextInt();
        int coupon = 0;

        switch (grade) {
            case 1 : 
                coupon = 1000;
                break;
            case 2 : 
                coupon = 2000;
                break;
            case 3 : 
                coupon = 3000;
                break;
            default : 
                coupon = 500;
                break;
        }
        System.out.println("발급받은 쿠폰 " + coupon);

        scan.close();
    }
}
