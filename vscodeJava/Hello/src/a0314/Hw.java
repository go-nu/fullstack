package a0314;

import java.util.Scanner;

public class Hw {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);

        int num1 = s.nextInt();
        int num2 = s.nextInt();
        int num3 = s.nextInt();
        
        int max = 0;

        if (num1 > num2) {
            if (num1 > num3) {
                max = num1;        
            } else {
                max = num3;
            }
        } else {
            if (num2 > num3) {
                max = num2;        
            } else {
                max = num3;
            }
        }

        System.out.println(max);

        s.close();
    }
}
