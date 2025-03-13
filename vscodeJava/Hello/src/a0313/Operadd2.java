package a0313;
public class Operadd2 {
    public static void main(String[] args) {
        int a = 1;
        int b = 0;
        b = ++a; // a 증가 후 b에 대입
        System.out.println("a = " + a + ", b = " + b);

        a = 1;
        b = 0;
        b = a++; // b 대입 후 a 증가
        System.out.println("a = " + a + ", b = " + b);
    }
}
