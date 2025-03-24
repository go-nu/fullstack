package a0324.업캐스팅;

public class UpCastingTest {
    public static void main(String[] args) {
        Square s = new Square();
        s.name = "정사각형";
        Triangle t = new Triangle();
        t.name = "삼각형";
        Circle c = new Circle();
        c.name = "원";

        Shape[] shapes = {(Shape)s, (Shape)t, (Shape)c}; // 부모의 타입으로 업 캐스팅 배열 생성
        
        for(int i = 0; i < shapes.length; i++) {
            System.out.printf("%d번 인덱스의 도형 %s\n", i, shapes[i].name);
        }
    }
}
// 상속 관계가 정의된 자식 객체는 부모의 타입으로 해석, 즉 업캐스팅될 수 있다.
// 서로 다른 자식 객체를 부모의 타입으로 묶어 관리할 수 있다.


// 부모클래스
class Shape {
    String name;
}
// 자식클래스
class Square extends Shape {}
class Triangle extends Shape {}
class Circle extends Shape {}