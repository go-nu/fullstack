package a0319;

public class Ch5_10_1 {
    public static void main(String[] args) {
        int[][] score = {
                { 100, 95, 46 },
                { 20, 20, 20 },
                { 30, 30, 30 },
                { 40, 40, 40 }
        };
        System.out.println("번호 국어 영어 수학 합계 평균");
        System.out.println("=============================");

        for (int i = 0; i < score.length; i++) {
            System.out.printf("%3d", i + 1); // 번호
            for (int j = 0; j < score[i].length; j++) {
                System.out.printf("%5d", score[i][j]); // 과목별 점수
            }
            System.out.printf("%5d  %.1f\n", sumRows(score, i), average(score, i)); // 합계 및 평균
        }

        System.out.println("=============================");

        System.out.print("총점:");
        for (int i = 0; i < score[i].length; i++) {
            System.out.printf("%4d ", sumColumns(score, i));
        }
        System.out.println();
        int n = score.length;
        System.out.println(n);

    }

    private static int sumRows(int arr[][], int n) { // 번호별 합계
        int sum = 0;
        for (int i = 0; i < arr[n].length; i++) {
            sum += arr[n][i];
        }
        return sum;
    }

    private static int sumColumns(int[][] arr, int n) { // 과목별 합계
        int sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum = sum + arr[i][n];
        }
        return sum;
    }

    private static double average(int[][] arr, int n) { // 가로 평균
        int sum = sumRows(arr, n);
        double average = (double) sum / arr[n].length;
        return average;
    }
}
