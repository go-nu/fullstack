package a0326.casting1;

public class AnimalMain {
    public static void main(String[] args) {
        Animal eagle = new Eagle();

        eagle.sleep();
        // eagle.eat();

        Eagle eagleObj = (Eagle)eagle; // 강제 형변환
        eagleObj.eat(); // Eagle 클래스의 eat() 메서드

        System.out.println(eagle instanceof Animal);
    }
}
