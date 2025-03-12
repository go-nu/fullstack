package a0312;

public class Var7 {
    public static void main(String[] args) {
        // 정수
        byte b = 127; // -128 ~ 127
        short s = 32767; // -32,768 ~ 32,767
        // 정수형 대표 int
        int i = 2147483647; //-2,157,483,648 ~ 2,157,483,647 (약 20억)
        
        //-9,223,372,036,854,775,808 ~ 9,223,372,036,854,775,807 + "L"(long은 뒤에 L 추가)
        long l = 9223372036854775807L;

        // 실수
        float f = 10.0f; // float 뒤에 f 추가
        // 실수형 대표 double
        double d = 10.0;

        // 변수명
        // int 1num = 10; 변수명은 숫자로 시작할 수 없음.
        // int int = 20; 예약어는 변수명으로 사용할 수 없음.
        // 변수명에는 영문자, 숫자, '$' 또는 '_'을 사용
        // 변수명은 소문자로 시작하는 것이 일반적, 이어지는 단어는 대문자로 시작
        // 낙타 표기법(camel case). (orderDetail, myAccount etc..)

        // 자바 정리
        // 클래스 이름은 대문자로 시작, 나머지는 소문자로 시작 + 낙타 표기법 적용
        // 클래스 : Person, OrderDetail
        // 변수 포함 나머지 : firstName, userAccount
        // 예외
        // 상수는 모두 대문자 사용 및 언더바(_)로 구분
        // USER_LIMIT
        // 패키지는 소문자
        // org.spring.boot
    }
}
