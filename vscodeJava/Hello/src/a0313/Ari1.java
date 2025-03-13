package a0313;

public class Ari1 {
    public static void main(String[] args) {
        byte v1 = 10; // 1byte
        byte v2 = 4;

        int v3 = 4; // 4byte
        long v4 = 10L; // 8byte

        int result1 = v1 + v2; // 모든 피연산자는 int 타입으로 자동변환
        System.out.println(result1);

        int result3 = v1 / v2;
        System.out.println(result3);

        double result4 = v1 / (double)v2; // v2를 double형으로 강제 변환
        System.out.println(result4);

        double result5 = 10 / 4; // 결과(result5)만 double
        System.out.println(result5);

        double result6 = 10 / (double)4;
        System.out.println(result6);

        double result7 = 10 / 4.0;
        System.out.println(result7);
    }
}
