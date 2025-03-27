package a0327.class1;

import java.util.Arrays;
import java.util.List;

public class Arrays1 {
        public static void main(String[] args) {
            int[] numbers = {5, 2, 8, 1, 3};
            for(int i = 0; i < numbers.length; i++) {
                System.out.print(numbers[i] + " ");
            }
            System.out.println("배열 내용" + Arrays.toString(numbers));
            // 배열을 문자열로 변환
            Arrays.sort(numbers);
            System.out.println("정렬된 배열: " + Arrays.toString(numbers));

            int index = Arrays.binarySearch(numbers, 5);
            System.out.println("값 5의 위치: " + index);

            int[] arr1 = {1, 2, 3};
            int[] arr2 = {1, 2, 3};
            int[] arr3 = {3, 2, 1};
            System.out.println("arr1과 arr2가 같은가? " + Arrays.equals(arr1, arr2));
            System.out.println("arr1과 arr2가 같은가? " + Arrays.equals(arr2, arr3));

            int[] numfill = new int [5];
            Arrays.fill(numfill, 7);
            System.out.println("배열 내용: " + Arrays.toString(numbers));

            int[] original = {1, 2, 3, 4, 5};
        
            int[] copy = Arrays.copyOf(original, 3); // 처음 3개 요소 복사
            System.out.println("복사된 배열: " + Arrays.toString(copy));

            int[] original1 = {10, 20, 30, 40, 50};
            int[] copy1 = Arrays.copyOfRange(original1, 1, 4);
            System.out.println("배열 내용: " + Arrays.toString(copy1));
            
            String[] fruits = {"Apple", "Banana", "Cherry"};
            List<String> fruitList = Arrays.asList(fruits);
            System.out.println("리스트 : " + fruitList);
        }
}
