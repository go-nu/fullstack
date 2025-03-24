package a0324.rpg2;

public class Con1 {
    public static void main(String[] args) {
        new BBB();
    }
}

class AAA {
    AAA() {
        System.out.println("AAA() 생성자 호출 완료");
    }
}
class BBB extends AAA{
    BBB() {
        super(); // 부모 생성자를 호출하는 코드, 생략되어있음.
        System.out.println("BBB() 생성자 호출 완료");
    }
}
// 상속 관계에서 자식 객체가 만들어지려면 부모영역이 먼저 완성
// 건물 1층이 완공되어야 2층을 올릴 수 있는것과 같은 이치
// 자식 생성자는 부모 생성자를 반드시 호출해야하는데, 이를 생략한 경우 부모 생성자 호출 코드
// super가 자동 생성됨.