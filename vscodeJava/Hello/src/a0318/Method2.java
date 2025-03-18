package a0318;

public class Method2 {
    public static void main(String[] args) {
        int sum1 = add(5,10);
        System.out.println("결과1 출력 : " + sum1);
        int sum2 = add(15,20);
        System.out.println("결과2 출력 : " + sum2);
        int sub1 = subtract(10,2);
        System.out.println("결과3 출력 : " + sub1);
        int mul1 = multiply(10,2);
        System.out.println("결과4 출력 : " + mul1);
        int div1 = divide(10,2);
        System.out.println("결과5 출력 : " + div1);
    }
        
    private static int add(int a, int b) {
        System.out.println(a +" + " + b + " 연산수행");
        int sum = a + b;

        return sum;
    }
    // abb, subtract, multiply, divide
    private static int subtract(int a, int b){
        System.out.println(a +" - " + b + " 연산수행");
        int sub = a - b;

        return sub;
    }
    private static int multiply(int a, int b){
        System.out.println(a +" * " + b + " 연산수행");
        int mul = a * b;

        return mul;
    }
    private static int divide(int a, int b){
        System.out.println(a +" / " + b + " 연산수행");
        int div = a / b;

        return div;
    }
}
