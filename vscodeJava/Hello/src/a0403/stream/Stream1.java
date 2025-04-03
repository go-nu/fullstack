package a0403.stream;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;

public class Stream1 {
    public static void main(String[] args) {
        int[] data = {5, 6, 4, 2, 3, 1, 1, 2, 2, 4, 8};
        // arrayList 생성 - 짝수만 포함
        ArrayList<Integer> dataList = new ArrayList<>();
        for(int i = 0; i < data.length; i++) {
            if (data[i] % 2 == 0) {
                dataList.add(data[i]);
            }
        }

        // Set을 사용한 중복제거
        HashSet<Integer> dataSet = new HashSet<>(dataList);
        
        // Set을 다시 ArrayList로 변환
        ArrayList<Integer> distinctList = new ArrayList<>(dataSet);
        
        // 역순 정렬
        distinctList.sort(Comparator.reverseOrder());
        // 순방향 정렬 distinctList.sort(Comparator.naturalOrder());

        // Integer List를 배열로 변환
        int[] result = new int[distinctList.size()];
        for(int i = 0; i < distinctList.size(); i++) {
            result[i] =  distinctList.get(i);
        }
        for(int num : result) {
            System.out.println(num + " ");
        }
    }
}
