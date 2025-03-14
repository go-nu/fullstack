package a0314;

import java.util.Scanner;

public class DistnaceEx {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);

        System.out.print("distance: ");
        int distance = s.nextInt();
        String ride;
        if (distance <= 1) {
            ride = "도보";
        } else if (distance <= 10) {
            ride = "자전거";
        } else if (distance <= 100) {
            ride = "자동차";
        } else {
            ride = "비행기";
        }

        System.out.println(ride + "를 이용하세요.");

        s.close();
    }
}
