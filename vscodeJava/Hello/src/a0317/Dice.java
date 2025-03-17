package a0317;

public class Dice {
    public static void main(String[] args) {
        int i = 0;
        int j = 0;
        int sum = 0;
        while (sum != 5) {
            // Math.random() 0~1사이 무작위 숫자 생성
            i = (int)(Math.random() * 6) + 1;
            j = (int)(Math.random() * 6) + 1;
            System.out.println(i + ", " + j);
            sum = i + j;
        }
    }
    
}
