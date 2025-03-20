package a0320;

import java.util.Scanner;

public class Ex01 {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        boolean flag = true;
        int[] scores = new int[0]; // 배열 선언
        int n = 0; // 배열 갯수로 사용할 변수 n 선언

        while (flag) {
            System.out.println("---------------------------------------------------");
            System.out.println("1.학생수 | 2.점수입력 | 3.점수리스트 | 4.분석 | 5.종료");
            System.out.println("---------------------------------------------------");
            System.out.print("선택> ");
            String select = s.nextLine();
            
            switch (select) {
                case "1":
                    System.out.printf("학생수> ");    
                    n = s.nextInt();
                    scores = new int[n]; // 배열 길이 초기화
                    break;
                case "2":
                    for(int i = 0; i < n; i++) {
                        System.out.printf("scores[%d]> ", i);
                        int k = s.nextInt();
                        scores[i] = k;
                    }
                    break;
                case "3":
                    for(int i = 0; i < n; i++) {
                        System.out.printf("scores[%d]> %d\n", i, scores[i]);
                    }
                    break;
                case "4" :
                    int max = 0;
                    int sum = 0;
                    float average = 0.0f;
                    for(int i = 0; i < n; i++) {
                        if(scores[i] > max) {
                            max = scores[i];
                        }
                        sum += scores[i];
                    }
                    average = (float)sum / scores.length;
                    System.out.printf("최고 점수: %d\n", max);
                    System.out.printf("평균 점수: %.1f\n", average);
                    break;
                case "5" :
                    flag = false;
                    System.out.println("프로그램 종료");
                    break;                    
                default:
                    break;
            }
        }

        s.close();

    }

}
