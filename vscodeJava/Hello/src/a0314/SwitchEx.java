package a0314;

import java.util.Scanner;

public class SwitchEx {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);

        int m = s.nextInt();
        String season = "";

        switch (m) {
            case 12:
            case 1:
            case 2:
                season = "winter";
                break;
            case 3:
            case 4:
            case 5:
                season = "spring";
                break;
            case 6:
            case 7:
            case 8:
                season = "summer";
                break;
            case 9:
            case 10:
            case 11:
                season = "fall";
                break;
        
            default:
                break;
        }
        
        System.out.println(m + "월은 " + season + "입니다.");

        s.close();
    }
}
