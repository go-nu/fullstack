package a0314;

import java.util.Scanner;

public class Switch2 {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        System.out.print("회원 등급 입력 : ");
        int grade = scan.nextInt();
        int coupon = 0;

        switch (grade) {
            case 1 : 
                coupon = 1000;
                break;
            case 2 : // 다른 조건에서 같은 내용을 불러올 때, break; 없이 이어 쓰기 가능
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
