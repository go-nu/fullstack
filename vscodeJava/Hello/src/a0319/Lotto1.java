package a0319;

import java.util.Arrays;

public class Lotto1 {
    public static void main(String[] args) {
        int[] ball = new int[45];
        for(int i = 0; i < ball.length; i++) {
            ball[i] = i + 1;
        }
        // System.out.println(Arrays.toString(ball));
        /*
        for(int i = 0; i < 1000; i++) {
            int temp = 0;
            int j = (int)(Math.random()*45);
            temp = ball[0];
            ball[0] = ball[j];
            ball[j] = temp;
        }
        System.out.println(Arrays.toString(ball));
        */
        for(int i = 0; i < 6; i++) {
            int temp = 0;
            int j = (int)(Math.random()*45);
            temp = ball[i];
            ball[i] = ball[j];
            ball[j] = temp;
        }
        // System.out.println(Arrays.toString(ball));
        
        for(int i = 0; i < 6; i++) {
            System.out.printf("ball[%d]=%d \n", i, ball[i]);
        }
    }
}
