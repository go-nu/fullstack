package a0314;

import java.util.Scanner;

public class Scanner3 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("정수 두개 입력 : ");
        int num1 = scanner.nextInt();
        int num2 = scanner.nextInt();

        // System.out.println(num1 + " " + num2);
        System.out.printf("%d %d", num1, num2);

        scanner.close();        
    }
}
