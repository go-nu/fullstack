package a0331.sort;

import java.util.Arrays;

public class InsertionSort {
    public static void main(String[] args) {
        // 앞에서부터 해당원소가 위치할 곳을 탐색하고 해당 위치에 삽입
        int[] arr = {63, 34, 25, 17, 22, 11, 90};
        insertionSort1(arr);
        System.out.println("Sorted Array : " + Arrays.toString(arr));
    }

    private static void insertionSort1(int[] arr) {
        int n = arr.length;

        for(int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i-1;
            while (j >= 0 && arr[j] > key) {
                arr[j+1] = arr[j];
                j--;
                System.out.println(Arrays.toString(arr));
            }
            arr[j+1] = key;
        }
    }
}
