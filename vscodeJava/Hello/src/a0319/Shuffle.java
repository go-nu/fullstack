package a0319;

import java.util.Arrays;

public class Shuffle {
    public static void main(String[] args) {
        int[] numArr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        /* 
        for(int i = 0; i < numArr.length; i++) {
            // System.out.println(numArr[i]);
            System.out.print(numArr[i] + " ");
        }
        for(int num:numArr) {
            System.out.println(num);
        }
        System.out.println(Arrays.toString(numArr));
         */
        for (int i = 0; i < 100; i++) {
            int j = (int)(Math.random()*10);
            int temp = numArr[0];
            numArr[0] = numArr[j];
            numArr[j] = temp;
        }
        System.out.println(Arrays.toString(numArr));
    }
}
