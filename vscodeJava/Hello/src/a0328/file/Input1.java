package a0328.file;

import java.io.IOException;
import java.io.InputStream;

public class Input1 {
    public static void main(String[] args) throws IOException {
        InputStream in = System.in;
        // System.in - 키보드로 입력 받겠다.
        // InputStream read - 1byte의 입력

        int a;
        a = in.read();
        System.out.println(a);
        // 입력받은 문자를 ASC코드로 변환
        // ASC코드는 7bit를 활용한 문자 표현 코드
        // 알파벳 대소문자, 숫자, 특수기호
        //  0 -> 48, a -> 97, A -> 65

    }
}
