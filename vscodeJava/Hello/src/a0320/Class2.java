package a0320;

public class Class2 {
    public static void main(String[] args) {
        Dog d1 = new Dog();
        Dog d2 = new Dog();

        // 객체 상태 변경
        d1.name = "망고";
        d1.breeds = "골드리트리버";
        d1.age = 2;

        d2.name = "까미";
        d2.breeds = "시고르잡종";
        d2.age = 3;
        
        System.out.printf("d1 => {%s, %s, %d세}\n", d1.name, d1.breeds, d1.age);
        System.out.printf("d2 => {%s, %s, %d세}\n", d2.name, d2.breeds, d2.age);
        d1.wag();
        d2.bark();
        d1.bark(3);
        // method overloading : 동일한 이름의 메소드를 입력변수의 차이로 구분
    }
}
class Dog {
    String name; // 필드(인스턴스 변수)
    String breeds;
    int age;

    void wag() {
        System.out.printf("[%s] 살랑살랑~\n", name);
    }
    void bark() {
        System.out.printf("[%s] 멍멍\n", name);
    }
    void bark(int times) {
        String sound = "컹컹";
        System.out.printf("[%s] %s(x%d)\n", name, sound, times);
    }
}