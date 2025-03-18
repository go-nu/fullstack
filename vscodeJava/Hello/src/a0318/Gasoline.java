package a0318;

public class Gasoline {
    public static void main(String[] args) {
        // 연비 = 이동거리 / 사용량
        double o = 8.86;
        double d = 182.736;
        System.out.printf("%fL를 충전한 자동차의 총 주행거리가 %f일때, 해당 자동차의 연비는 %f", o, d, yb(o,d));
    }
    private static double yb(double distance, double oil) {
        
        return distance / oil;
    }
}
