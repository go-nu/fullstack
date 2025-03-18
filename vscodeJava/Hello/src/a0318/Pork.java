package a0318;

public class Pork {
    public static void main(String[] args) {
        int p = 3;
        double k = kcal(p);
        System.out.printf("삼겹살 %d인분의 칼로리: %.2f kcal", p, k);
    }
    private static double kcal(int n) {
        double total = (double)n * 180 * 5.179;
        return total;
    }
}
