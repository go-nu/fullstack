package a0321.Account1;

public class AvengerTest {
    public static void main(String[] args) {
        // 객체 생성 및 호출을 동시에
        Avenger ironMan = new Avenger("아이언맨", 100, 120);
        Avenger hulk = new Avenger("아이언맨", 200, 80);
        // 객체 정보 출력
        System.out.println(ironMan.toStr());
        System.out.println(hulk.toStr());
    }
}
class Avenger {
    private String name;
    private int power;
    private int speed;

    public Avenger(String name, int power, int speed) {
        this.name = name;
        this.power = power;
        this.speed = speed;
    }

    public String toStr() {
        return String.format("Avenger{name : %s, power : %d, speed : %d}", name, power, speed);
    }

    
}