package a0324;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

public class ArrayList1 {
    public static void main(String[] args) {
        // 컬렉션
        ArrayList<Integer> arrList = new ArrayList<Integer>();

        // add 메소드를 이용한 요소의 저장
        arrList.add(40);
        arrList.add(20);
        arrList.add(30);
        arrList.add(10);

        // for문과 get()을 이용한 요소 출력
        for(int i = 0; i < arrList.size(); i++) {
            System.out.print(arrList.get(i) + " ");
        }
        System.out.println();
        // Enhance for 향상된 for문
        for(int e : arrList) {
            System.out.print(e + " ");
        }
        System.out.println();
        // Collections.sort() 메소드를 이용한 요소 정렬
        Collections.sort(arrList);
        for(int e : arrList) {
            System.out.print(e + " ");
        }
        System.out.println();
        // iterator() 메소드와 get() 메소드를 이용한 요소 출력
        Iterator<Integer> iter = arrList.iterator();
        while(iter.hasNext()) {
            System.out.print(iter.next() + " ");
        }
        System.out.println();
        // set() 메소드를 이용한 요소 
        arrList.set(0, 20);

        for(int e : arrList) {
            System.out.print(e + " ");
        }
        System.out.println();
        // remove();
        arrList.remove(1);
        for(int e : arrList) {
            System.out.print(e + " ");
        }
        System.out.println();
        // 특정 인덱스 위치에 요소 삽입
        arrList.add(0, 10);
        for(int e : arrList) {
            System.out.print(e + " ");
        }
        System.out.println();
        // size() 메소드를 이요한 요소의 총 개수
        System.out.println("리스트의 크기 : " + arrList.size());
        

    }
}
