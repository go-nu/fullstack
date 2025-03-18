package a0318;
public class Array1 {
    public static void main(String[] args) {
        int[] numbers = {1, 2, 3, 4, 5};
        System.out.println(sumArray(numbers));
    }
    public static int sumArray(int[] arr) {
        int sum = 0;
        // for (int i=0; i<arr.length; i++) {
        //     sum += arr[i];
        // }

        for(int num:arr){
            sum += num;
        }
        return sum;
    }
}
