package a0327.exception1;

public class ExceptionEx03 {
    public static void main(String[] args) {
        try {
            int[] arr = new int[2];
            arr[3] = 10;
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("예외 발생: " + e.getMessage());
        } finally {
            System.out.println("finally 블록, 무조건 실행");
        }
    }
}
