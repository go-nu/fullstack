package a0312;

public class Temperature {
    public static void main(String[] args) {
        double fTemp = 77.0;
        double cTemp = (fTemp - 32.0) / 1.8;

        System.out.printf("화씨 %.1f도는 섭씨로 %.1f도입니다.", fTemp, cTemp);
    }
}
