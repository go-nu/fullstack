package a0331.sort;

import java.util.Arrays;

public class SelectionSort {
    public static void main(String[] args) {
        // 앞에서부터 해당원소가 위치할 곳을 탐색하고 해당 위치에 삽입
        int[] arr = {63, 34, 25, 17, 22, 11, 90};
        selectionSort1(arr);
        System.out.println("선택 정렬 : " + Arrays.toString(arr));
    }

    private static void selectionSort1(int[] arr) {
        int n = arr.length;

        for(int i = 0; i < n-1; i++) {
           int minIndex = i;
           for(int j = i+1; j < n; j++) {
                if(arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            System.out.println(Arrays.toString(arr));
            int temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
    }
}
