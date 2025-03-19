package a0319;

public class ArrEx2 {
    public static void main(String[] args) {
        String[] name = {"Elena", "Suzie", "John","Emily", "Neda", "Kate", "Alex", "Daniel", "Sam"};
        int[] score = {65, 74, 23, 75, 68, 96, 88, 98, 54};
        
        int max = 0;
        int count = 0;
        for(int i = 0; i < score.length; i++) {
            if (score[i] > max) {
                max = score[i];
                count = i;
            }
        }
        System.out.printf("1등: %s(%d)", name[count], max);
    }
}
