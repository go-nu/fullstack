package a0320;

public class Sum1 {
    public static void main(String[] args) {
        int[][] score = {
            {100, 95, 46}, 
            {20, 20, 20}, 
            {30, 30, 30}, 
            {40, 40, 40}
        };
        int kor = 0;
        int eng = 0;
        int math = 0;

        System.out.println("번호  국어  영어  수학  합계  평균");
        System.out.println("================================");
        
        for(int i = 0; i < score.length; i++) {
            int sum = 0;
            float avg = 0.0f;

            // 과목별 총점
            kor += score[i][0];
            eng += score[i][1];
            math += score[i][2];

            System.out.printf("%d", i + 1);

            for(int j = 0; j < score[i].length; j++) {
                // 가로합계
                sum += score[i][j];
                // 과목별 점수 출력
                System.out.printf("%4d", score[i][j]);
            }

            // 가로 평균 구하기, 가로 합계ㆍ평균 출력
            avg = (float)sum / score[i].length;
            System.out.printf("%4d %4.1f\n", sum, avg);
        }
        System.out.println("================================");
        System.out.printf("총점 %4d %4d %4d", kor, eng, math);
    }
}
