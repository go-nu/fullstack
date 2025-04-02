package a0402.optional;

import java.util.Optional;

// 값이 존재하면 작업을 수행(ifPresent(동작))
public class Optional3 {
    public static void main(String[] args) {
        Optional<String> name = Optional.of("Charlie");

        // 값이 있으면 동작 수행
        name.ifPresent(n -> System.out.println("Hello " + n));

        Optional<String> emptyName = Optional.ofNullable(null);
        emptyName.ifPresent(n -> System.out.println("Hello, " + n));  // 출력: 아무 것도 출력되지 않음
    }
}
