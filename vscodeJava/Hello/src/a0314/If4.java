package a0314;

import java.util.Scanner;

public class If4 {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        System.out.print("회원 등급 입력 : ");
        int grade = scan.nextInt();
        
        if (grade == 1) {
            System.out.println("발급받은 쿠폰 1000");
        } else if (grade == 2) {
            System.out.println("발급받은 쿠폰 2000");
        } else if (grade == 3) {
            System.out.println("발급받은 쿠폰 3000");
        } else {
            System.out.println("발급받은 쿠폰 500");
        }

        scan.close();
    }
}
