package a0402.optional;

import java.util.Optional;

public class Optional5 {
    public static void main(String[] args) {
        Optional<String> optionalName = findName("Mark");
        System.out.println(optionalName.orElse("Name not found"));
    }

    private static Optional<String> findName(String name) {
        if("John".equals(name)) {
            return Optional.of(name);
        }
        return Optional.empty();
    }
}
