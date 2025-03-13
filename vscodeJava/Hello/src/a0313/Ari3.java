package a0313;

public class Ari3 {
    public static void main(String[] args) {
        String strSeven = "7";
        String strPi = "3.14";

        int a = Integer.parseInt(strSeven); // 문자를 정수로 변환
        double b = Double.parseDouble(strPi); // 문자를 실수로 변환

        double c = a + b;
        System.out.printf("%d + %.2f = %.2f", a, b, c);
    }
}
