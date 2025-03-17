package a0317;

import java.util.Scanner;

public class Tree {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        System.out.print("줄 수 입력 : ");
        int rows = s.nextInt();
        for(int i = 0; i < rows; i++) {
            for(int j = rows; j >= i; j--) {
                System.out.print(" ");
            }
            for(int k = 1; k <= (2*i+1); k++) {
                System.out.print("*");
            }
            System.out.println();
        }
        s.close();
    }
}
