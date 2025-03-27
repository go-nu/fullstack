package a0327.exception1;

public class customException extends Exception {
    public customException(String message) {
        super(message);
    }

}
// customException은 Exception(자바에서 최상의 예외)를 상속
// 사용자 정의 예외
// 생성자를 통해 예외 메세지를 전달할 수 있음.