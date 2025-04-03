package a0403.collection;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

// collection
// 1. 데이터를 효율적으로 저장
// 2. 배열보다 유연하게 크기를 변경할 수 있음
// 3. 자료구조 : List, Map, Set
// 4. ArrayList<String> - 제네릭지원으로 타입 안정성 제공
// 5. 다양한 구현체 제공: 성능과 특징에 맞게 선택가능

// List(순서 O, 중복 O)
// ArrayList 배열 기반, 빠른 검색 / 비효율적인 삽입, 삭제
// LinkedList 연결 List 기반, 빠른 삽입, 삭제 / 느린 검색
// Vector ArrayList와 유사, Thread 안전
// Stack LIFO 구조 push() - 삽입, pop() - 삭제
// Queue FIFO 구조
// Set(순서 X, 중복 X)
// HashSet Hash기반, 순서 X, 중복 X
// LinkedHashSet 입력 순서 유지, 중복 X
// TreeSet 정렬된 상태 유지(오름차순)

// Map(K,V) Key 중복 X
// HashMap(키 순서 X, 빠른 검색)
// LinkedHashMap 입력 순서 유지
// TreeMap 정렬된 상태 유지(Key 기준)

public class Set1 {
    public static void main(String[] args) {
        // 객체 선언
        Set<String> set = new HashSet<String>();
        // 데이터 삽입 (.add)
        set.add("apple");
        set.add("banana");
        set.add("pyopyo");
        set.add("kiwi");
        for(String e : set) {
            System.out.println(e);
        }
        System.out.println();
        // 데이터 삭제 (.remove)
        set.remove("apple");
        set.add("orange");
        set.add("orange"); // 중복 요소의 저장 X
        Iterator<String> iterSet = set.iterator();
        while (iterSet.hasNext()) {
            System.out.println(iterSet.next() + " ");
        }
        System.out.println();
        System.out.println(set);

        System.out.println(set.contains("orange"));

        // 전체 데이터 삭제
        set.clear();
        System.out.println(set);
        System.out.println(set.isEmpty()); // set이 비어있는지 확인

    }
}
