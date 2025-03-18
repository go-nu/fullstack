package a0318;

public class Ex1 {
    public static void main(String[] args) {
        System.out.println(max(10, 20));
        System.out.println(max(5, 3));
    }
    public static int max(int a, int b) {
        if(a > b) {
            return a;
        }
        else return b;
    }
}
