package a0318;

public class RandomNumber {
    public static void main(String[] args) {
        int n = rollDice();
        System.out.printf("주사위의 눈 : %d", n);

    }

    public static int rollDice() {
        // int result = (int)(Math.random() * 6) + 1;
        
        // return result;
    
        double x = (Math.random() * 6);
        int temp = (int) x;
        
        return temp + 1;
    }
}
