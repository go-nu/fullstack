package a0313;

public class Ari2 {
    public static void main(String[] args) {
        double tall = 176.6;
        double weight = 74.34;

        System.out.printf("신장 %.1fcm\n", tall);
        System.out.printf("체중 %.1fkg\n", weight);

        // 다운 캐스팅
        System.out.printf("신장 %dcm\n", (int)tall);
        System.out.printf("체중 %dkg\n", (int)weight);
    }
}
