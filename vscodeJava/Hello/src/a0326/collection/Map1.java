package a0326.collection;

import java.util.HashMap;
import java.util.Iterator;

public class Map1 {
    public static void main(String[] args) {
        HashMap<String, Integer> hm = new HashMap<String, Integer>();

        // put() 메소드를 이용한 요소 초기화
        hm.put("삼십", 30);
        hm.put("십", 10);
        hm.put("사십", 40);
        hm.put("이십", 20);

        // 순서가 보정되지 않는 키값 출력
        System.out.println("맵의 저장된 키들의 집합 : " + hm.keySet());
        // for 문으로 get() 메소드를 이용한 요소 출력
        for(String key : hm.keySet()) {
            System.out.printf("키 : %s, 값 : %d\n", key, hm.get(key));
        }
        System.out.println();

        // remove() 메소드를 이용한 요소 제거
        hm.remove("사십");

        // iterator() 메소드와 get() 메소드를 이용한 요소 출력
        Iterator<String> keys = hm.keySet().iterator();
        while (keys.hasNext()) {
            String key = keys.next();
            System.out.printf("키 : %s, 값 : %d\n", key, hm.get(key));
        }
        System.out.println();

        // replace() 메소드를 이용한 값 수정
        hm.replace("이십", 200);
        for(String key : hm.keySet()) {
            System.out.printf("키 : %s, 값 : %d\n", key, hm.get(key));
        }
        System.out.println();

        // size() 메소드를 이용한 요소 갯수
        System.out.println("맵의 크기 : " + hm.size());
    }
}
