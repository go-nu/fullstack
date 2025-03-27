package a0327.exception1;

public class ExceptionEx04 {
    public static void main(String[] args) {
        try {
            checkNumber(-5);
        } catch(customException e) {
            System.out.println("예외 발생: " + e.getMessage());
        }
    }
    // 1. try 내부 checkNumber() 호출 > 음수전달 -> 예외발생
    // 2. catch(customException e)에서 예외를 잡아 예외 메세지 출력.

    private static void checkNumber(int num) throws customException {
        if(num < 0) throw new customException("음수는 허용 않됩니다.");
        System.out.println("입력 값: " + num);
    }
    // 매개변수 num이 음수일 경우, customException 발생.
    // throws customException을 선언하여, 이 메세지를 예외로 던진다.
}
