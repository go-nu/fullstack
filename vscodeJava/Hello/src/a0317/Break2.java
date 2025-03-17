package a0317;

public class Break2 {
    public static void main(String[] args) {
        int sum = 0;
        int i = 1;
        for (;;) {
            sum += i;
            if (sum > 10) {
                System.out.println("합이 10보다 커지면 종료, i = " + i + " sum = " + sum);
                break;
            }
            i++;
        }
    }
    // while(true) for(;;) 무한루프
}
