package a0401.lambda;

import java.util.Arrays;
import java.util.List;

public class Lambda2 {
    public static void main(String[] args) {
        List<String> names = Arrays.asList("banana", "apple", "orange");
        // 이름 기준으로 내림차순 정렬
        names.sort((s1, s2) -> s2.compareTo(s1));
        System.out.println("내림차순 정렬 : " + names);
        names.sort((s1, s2) -> s1.compareTo(s2));
        System.out.println("오름차순 정렬 : " + names);
        
        // 람다
        names.sort(String::compareTo);
        System.out.println("오름차순 정렬 : " + names);
        // s2.compareTo(s1) 대신 메서드 참조(::연산자)를 사용하여 다 간결하게 표현
        
    }
}
