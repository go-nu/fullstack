package a0318;

public class Method3 {
    public static void main(String[] args) {
        int num1 = 10;
        int num2 = 5;
        System.out.println("덧셈" + Calculator.add(num1, num2));
        System.out.println("뺄셈" + Calculator.subtract(num1, num2));
        System.out.println("곱셈" + Calculator.multiply(num1, num2));
        System.out.println("나눗셈" + Calculator.divide(num1, num2));
    }
}
class Calculator {
    public static int add(int a, int b) {
        return a + b;
    }
    public static int subtract(int a, int b) {
        return a - b;
    }
    public static int multiply(int a, int b) {
        return a * b;
    }
    public static double divide(int a, int b) {
        if(b == 0) {
            throw new ArithmeticException("0으로 나눌 수 없습니다.");
        }
        return (double)a / (double)b;
    }
}