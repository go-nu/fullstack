package a0320;

public class Sum2 {
    public static void main(String[] args) {
        int[][] score = {
                {100, 95, 46},
                {20, 20, 20},
                {30, 30, 30},
                {40, 40, 40}
        };

        printScores(score);
    }

    public static void printScores(int[][] score) {
        int kor = 0, eng = 0, math = 0;
        System.out.println("번호  국어  영어  수학  합계  평균");
        System.out.println("======================================");

        for (int i = 0; i < score.length; i++) {
            int sum = 0;
            System.out.printf("%d", i + 1);
            
            for (int j = 0; j < score[i].length; j++) {
                sum += score[i][j];
                System.out.printf("%5d", score[i][j]);
            }
            
            float avg = sum / (float) score[i].length;
            System.out.printf("%5d %5.1f%n", sum, avg);
            
            kor += score[i][0];
            eng += score[i][1];
            math += score[i][2];
        }
        
        System.out.println("=============================");
        printTotalScores(kor, eng, math);
    }

    public static void printTotalScores(int kor, int eng, int math) {
        System.out.printf("총점: %4d %4d %4d%n", kor, eng, math);
    }
}
