package a0314;

import java.util.Scanner;

public class If1_1 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("나이를 입력하세요 : ");
        int age = scanner.nextInt();

        if(age >= 20) 
            System.out.println("당신은 성인입니다.");
        else 
            System.out.println("당신은 미성년자입니다.");

        scanner.close();        
    }
}
