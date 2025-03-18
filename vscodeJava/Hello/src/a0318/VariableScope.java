package a0318;

public class VariableScope {
    public static void main(String[] args) {
        int x = 5;
        System.out.println(x); // 5
        print(x);
        System.out.println(x); // 5
    }
    public static void print(int x) {
        System.out.println(x); // 5
        x += 10;
        System.out.println(x); // 15
    }
}
