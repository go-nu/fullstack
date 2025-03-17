package a0317;
public class Break1 {
    public static void main(String[] args) {
        // 1부터 누적해서 더하다가 합이 10을 넘어가는 처음 수
        int sum = 0;
        int i = 1;
        while (true) {
            sum += i;
            if (sum > 10) {
                System.out.println("합이 10보다 커지면 종료, i = " + i + " sum = " + sum);
                break;
            }
            i++;
        }
    }
    // while (true)이기에 무한 반복이 된다. -> break;를 통해 탈출
}
