package a0317;

public class WhileEx1 {
    public static void main(String[] args) {
        int count = 0;
        while (count < 10) {
            count += 1;
            System.out.println(count);
        }
        for(int i = 1; i <= 10; i++) {
            count = i;
            System.out.println(count);
        }
    }
}
