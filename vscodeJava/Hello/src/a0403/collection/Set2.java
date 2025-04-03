package a0403.collection;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Set2 {
    public static void main(String[] args) {
        List<String> items = new ArrayList<>();

        // 중복 값이 있는 리스트
        items.add("apple");
        items.add("banana");
        items.add("orange");
        items.add("apple");
        items.add("grape");

        // HashSet을 사용한 중복제거
        Set<String> uniqueItems = new HashSet<>(items);
        System.out.println("중복 제거 후 리스트: " + uniqueItems);
    }
}
