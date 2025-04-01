package a0401.lambda;

import java.util.Arrays;
import java.util.List;

public class Lambda1 {
    public static void main(String[] args) {
        // List<String> name = new ArrayList<>();
        // name.add("apple");
        // name.add("banana");
        // name.add("orange");      
        // for(int i = 0 ; i < name.size(); i++) {
        //     System.out.println(name.get(i));
        // }

        List<String> names = Arrays.asList("apple", "banana", "orange");
        for(String name : names) {
            System.out.println(name);
        }
        System.out.println();

        // 람다 forEach
        names.forEach(name -> System.out.println(name));
        System.out.println();

        // 스트림
        // 길이가 6이상(글자의 갯수) 요소만 필터링해라.
        names.stream()
            .filter(name -> name.length() >= 6) // 6글자 이상의 요소만 필터링
            .forEach(System.out::println); // 필터링된 요소들 출력

    }
}
