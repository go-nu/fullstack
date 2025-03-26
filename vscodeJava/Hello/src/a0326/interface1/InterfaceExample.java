package a0326.interface1;

interface Animal {
    void makeSound(); // 추상 메서드(abstract 샹략 가능)
    // 일반 메서드 사용 금지
}
// 인터페이스를 구현하는 클래스
class Dog implements Animal {

    @Override
    public void makeSound() {
        System.out.println("멍멍");
    }

}

class Cat implements Animal {

    @Override
    public void makeSound() {
        System.out.println("애옹");
    }

}


public class InterfaceExample {
    public static void main(String[] args) {
        // Animal ani = new Animal();
        // 인터페이스는 추상클래스 같이 자신을 객체로 만들 수 없음.
        Animal dog = new Dog();
        Animal cat = new Cat();
        // main() 메서드에서 다형성을 활용하여 Animal 타입으로 자식 객체를 사용할 수 있음.
        dog.makeSound();
        cat.makeSound();
    }
}
