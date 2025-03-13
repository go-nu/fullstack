package a0313;

public class Logical1 {
    public static void main(String[] args) {
        System.out.println("&&: AND 연산");
        System.out.println(true && true);
        System.out.println(true && false);
        System.out.println(false && false);

        System.out.println("||: OR 연산"); // 여러개 중 하나만 true면 true
        System.out.println(true || true);
        System.out.println(true || false);
        System.out.println(false || false);
        
        System.out.println("!: NOT 연산");
        System.out.println(!true);
        System.out.println(!false);

        System.out.println("변수 활용");
        boolean a = true;
        boolean b = false;
        System.out.println(a && b); // f
        System.out.println(a || b); // t
        System.out.println(!a); // f
        System.out.println(!b); // t
    }
}
