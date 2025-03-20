package a0320;

public class Class3 {
    public static void main(String[] args) {
        // Card.width; Card.height; 클래스 변수는 객체 생성 없이 사용 가능
        System.out.println("Card.width = " + Card.width);
        System.out.println("Card.height = " + Card.height);

        // 객체 생성, 참조변수 c1
        Card c1 = new Card(); // c1의 주소를 할당
        c1.kind = "Heart";
        c1.number = 7;
        
        Card c2 = new Card();
        c2.kind = "Spade";
        c2.number = 5;

        System.out.println("c1은 " + c1.kind + ", " + c1.number + "이며, 크기는 " + c1.width + ", " + c1.height);
        
        System.out.println("c1의 width와 height를 50, 80으로 변경");
        
        // c1.width = 50;
        // c1.height = 80;
        // 클래스 변수에 대해서는 참조변수보다 클래스를 직접 선언하는게 좋음
        Card.width = 80;
        Card.height = 80;

        System.out.println("c1은 " + c1.kind + ", " + c1.number + "이며, 크기는 " + c1.width + ", " + c1.height);
        System.out.println("c2는 " + c2.kind + ", " + c2.number + "이며, 크기는 " + c2.width + ", " + c2.height);
    }
}
class Card {
    String kind;
    int number;
    // 클래스 내부 값이 고정적인 요소에 대해서는 static을 사용해서 고정(fix)한다.
    // 클래스 변수, 객체 생성 없이 사용 가능
    static int width = 100;
    static int height = 80;
}