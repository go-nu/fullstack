package a0320;

public class Class5_1 {
    public static void main(String[] args) {
    Person1 p1 = new Person1("홍길동", 31);
    Person1 p2 = new Person1();

    p2.name = "이순신";
    p2.age = 25;

    System.out.printf("이름: %s, 나이 %d세\n", p1.name, p1.age);
    System.out.printf("이름: %s, 나이 %d세", p2.name, p2.age);
    }
}

class Person1 {
    String name;
    int age;

    public Person1(String n, int a){
        name = n;
        age = a;
    }

    // 프로그램에서 기본 생성자를 만들어준다.
    // 둘 다 사용하는 경우에는 기본 생성자를 직접 만들어 줘야함
    public Person1(){

    }
}