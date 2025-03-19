package a0319;

public class Sum1 {
    public static void main(String[] args) {
        int sum = 0;
        float average = 0f;
        int[] score = {100, 88, 100, 100, 90};
        // sum과 average를 구하시오
        for(int i = 0; i < score.length; i++) {
            sum += score[i];
        }
        average = (float)sum / score.length;

        System.out.printf("총합은 %d이고 평균은 %.2f이다.", sum, average);
    }
}
