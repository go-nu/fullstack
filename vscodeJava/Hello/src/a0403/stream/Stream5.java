package a0403.stream;

import java.util.stream.Stream;

public class Stream5 {
    public static void main(String[] args) {    
        // 가변 메개변수에서 Stream 생성
        Stream<Double> stream = Stream.of(4.2, 2.5, 3.1, 1.5);
        stream.forEach(System.out::println);

        // of()를 사용하면 가변 매게변수에서 Stream 생성

    }
}