package a0317;
public class Break1_1 {
    public static void main(String[] args) {
        int sum = 0;
        int i = 1;
        boolean flag = true;
        while (flag) {  
            sum += i;
            if (sum > 10) {
                System.out.println("합이 10보다 커지면 종료, i = " + i + " sum = " + sum);
                flag = false;
            }
            i++;
        }
    }
}
