package a0324.rpg2;

public class CarTest {
    public static void main(String[] args) {
        Truck t = new Truck("봉고", 1.5);
        
    }
}
class Car {
    String name;

    public Car(String _name) {
        name = _name;
        System.out.printf("모델명: %s", name);
    }
    void move() {
        System.out.printf("[%s]가 이동합니다.\n, name");
    }
}

class Truck extends Car {
    double payload; // 트럭 최대 적재량(톤)

    public Truck(String _name, double _payload) {
        super(_name); // 부모생성자 호출
        payload = _payload;
    }
    void load() {
        System.out.printf("[%s]가 물건을 적재합니다. (최대 %.1ft)\n", name, payload);
    }
}