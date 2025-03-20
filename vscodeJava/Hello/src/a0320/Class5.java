package a0320;

public class Class5 {
    public static void main(String[] args) {
    Person person1 = new Person();
    person1.name = "홍길동";
    person1.age = 31;
    System.out.printf("이름: %s, 나이 %d세", person1.name, person1.age);
    }
}

class Person {
    String name;
    int age;

}