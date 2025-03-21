package a0321;

public class MyMath1 {
    public static void main(String[] args) {
        // int b = MyMath.add(5, 6);
        // 2. MyMath 객체 생성
        MyMath mm = new MyMath();
        // 3. MyMath 객체 mm사용
        int m1 = mm.add(5, 6);
        int m2 = mm.subtract(9, 5);
        int m3 = mm.multiply(4, 5);
        double m4 = mm.divide(8, 2);
        int m5 = mm.max(15, 6);
        System.out.printf("m1 += %d", m1);
        System.out.printf("\nm2 -= %d", m2);
        System.out.printf("\nm3 *= %d", m3);
        System.out.printf("\nm4 /= %.1f", m4);
        System.out.printf("\nm5 >= %d", m5);

    }
}

// 사칙연산을 수행하는 method를 가진 MyMath class
// method는 class 영역에 정의
// 1. MyMath class 생성
class MyMath {
    int add(int a, int b) {
        int result = a + b;
        return result;
    }
    int subtract(int a, int b) {
        return a - b;
    }
    int multiply(int a, int b) {
        return a * b;
    }
    int divide(int a, int b) {
        return a / b;
    }

    // 큰수 구하기
    int max(int a, int b) {
        if(a > b) return a;
        else return b;
    }
}