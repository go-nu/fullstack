package a0319;

public class MaxMin1 {
    public static void main(String[] args) {
        int[] score = {79, 88, 91, 33, 100, 55, 95};
        int max = 0;
        int min = 100;
        for(int i = 0; i < score.length; i++) {
            if(score[i] > max) {
                max = score[i];
            }
            if (score[i] < min) {
                min = score[i];
            }
        }
        System.out.println("max : " + max);
        System.out.println("min : " + min);
    }
}
