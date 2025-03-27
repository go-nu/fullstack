package a0327.package1;

public class Parent {
    protected String message = "나는 protected 변수입니다.";

    protected void showMessage() {
        System.out.println("Protected 메서드 실행: " + message);
    }
}
