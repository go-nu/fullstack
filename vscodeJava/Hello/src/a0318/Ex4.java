package a0318;

public class Ex4 {
    public static void main(String[] args) {
        System.out.println(factorial(5));
        System.out.println(factorial(3));
    }
    public static int factorial(int n) {
        // int result = 1;
        // for(int i = n; i > 0; i--){
        //     result *= i;
        // }

        if(n==0 || n ==1) {
            return 1;
        }
        return n * factorial(n-1);
    }
}
