package a0314;

import java.util.Scanner;

public class Scanner4 {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);

        System.out.print("실수 입력 : ");
        float num = s.nextFloat();

        num = (num + 0.005f) * 100;
        int i = (int)num;
        num  = (float)i / 100;

        System.out.printf("%.2f", num);

        s.close();
    }
}
