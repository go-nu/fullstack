package a0326.interface1;

interface Greeting {
    void sayHello();

    // java 8 이상부터 사용가능한 default method
    default void sayGoodbye(){
        System.out.println("안녕히 가세요");
    }

}
class Person implements Greeting {

    @Override
    public void sayHello() {
        System.out.println("안녕하세요");
    }

    // Person 클래스에만 존재하는 추가 메서드
    public void introduce() {
        System.out.println("저는 Person class입니다.");
    }

}

public class DefaultMethodEx {
    public static void main(String[] args) {
        Person person = new Person();
        person.sayHello();
        person.sayGoodbye(); // 인터페이스의 default method
        person.introduce();

        Greeting person1 = new Person();
        person1.sayHello();
        person1.sayGoodbye();
        // person1.introduce(); 컴파일 에러, Greeting 타입에는 introduce()가 없어 사용이 안됨.

        // 다운캐스팅 : 부모 -> 자식 타입 변환
        if(person1 instanceof Person) { // 부모 자식관계인지 물어보고
            ((Person)person1).introduce();
            // person1은 실제로는 Person의 객체이므로, Person으로 캐스팅하면 introduce() 호출가능.
        }

    }
}
