package a0401;

import java.util.Arrays;
import java.util.stream.Stream;

public class Ex2 {
    public static void main(String[] args) {
        String[] arr = new String[] {"a", "b", "c", "d", "e", "f"};

        Stream<String> stream1 = Arrays.stream(arr);
        // stream1.forEach(s -> System.out.print(s + " "));
        stream1.forEach(System.out::println);
    }
}
