package a0320;

public class ClassEx1 {
    public static void main(String[] args) {
        Square s = new Square();
        s.length = 4;
        System.out.printf("한 변의 길이가 %d인 정사각형의 넓이: %d", s.length, s.area());
    
    }
}

class Square {
    int length;
    int area() {
        return length*length;
    }
}
// 객체지향프로그래밍 Object-Oriented Programming OOP
// 장점
// 1. 프로그램 유지보수가 좋음
// 2. 코드의 재사용이 수월함

// 클래스 -> 객체의 설계도
// 클래스를 바탕으로 만들어진 프로그램의 구성요소가 객체
// 클래스 설계 - 필드와 메소드로 이루어짐
// Dog d1 = new Dog(); Dog객체를 만들어 d1과 연결한다.
// 클래스 변수 = new 클래스();