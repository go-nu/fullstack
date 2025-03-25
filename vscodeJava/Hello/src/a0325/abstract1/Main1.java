package a0325.abstract1;

public class Main1 {
    public static void main(String[] args) {
        Animal d = new Dog("buddy");
        d.makeSound();
        d.eat();
        // Animal 클래스가 추상클래스이므로, 직접 인스턴스를 만들 수 없어, 
        // Dog와 같은 하위 클래스에서 인스턴스를 만들어야함.

        Animal c = new Cat("나비");
        c.makeSound();
        c.eat();
    }
}
