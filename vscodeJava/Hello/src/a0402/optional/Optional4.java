package a0402.optional;

import java.util.Optional;

public class Optional4 {
    public static void main(String[] args) {
        Optional<String> emptyName = Optional.ofNullable(null);

        try {
            String name = emptyName.orElseThrow(() -> new IllegalArgumentException("name is Required"));
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
        }
        
        // 값이 있는 경우
        Optional<String> name = Optional.of("John");
        String validName = name.orElseThrow(() -> new IllegalArgumentException("Name is required"));
        System.out.println("Valid name: " + validName);  // 출력: Valid name: John
    }
}
