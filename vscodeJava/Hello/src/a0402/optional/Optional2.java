package a0402.optional;

import java.util.Optional;

// 값이 존재하면 처리하고, 없으면 Guest와 같은 기본 값 제공
public class Optional2 {
    public static void main(String[] args) {
        Optional<String> name = Optional.ofNullable("Alice");
        // 값이 있으면 값을 출력하고, 없으면 기본 값 출력
        String result = name.orElse("Guest");
        System.out.println("Hello " + result);
        
        // 값을 null로 설정하여 기본값 확인
        Optional<String> name1 = Optional.ofNullable(null);
        String result1 = name1.orElse("Guest");
        System.out.println("Hello " + result1);

    }
}
