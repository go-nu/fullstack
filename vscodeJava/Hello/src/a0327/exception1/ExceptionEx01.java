package a0327.exception1;

public class ExceptionEx01 {
    public static void main(String[] args) {
        try {
            int result = 10 / 0;
            System.out.println("결과: " + result);

        } catch (ArithmeticException e) {
            System.out.println("예외발생: " + e.getMessage());
        }
    }
}
