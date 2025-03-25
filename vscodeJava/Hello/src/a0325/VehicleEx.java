package a0325;

abstract class Vehicle {
    String name;

    public Vehicle(String name) {
        this.name = name;
    }

    abstract void move();
    void displayInfo() {
        System.out.println("이 차량은 [" + name + "]입니다.");
    }
}

class Car extends Vehicle {

    public Car(String name) {
            super(name);
        }
    
    @Override
    void move() {
        System.out.println("자동차 [" + name + "]이(가) 도로를 달립니다.");
    }
    
}

class Bicycle extends Vehicle {
    
    public Bicycle(String name) {
        super(name);
    }

    @Override
    void move() {
        System.out.println("자전거 [" + name + "]이(가) 도로를 달립니다.");
    }
}

public class VehicleEx {
    public static void main(String[] args) {
        Vehicle[] v = new Vehicle[2];
        v[0] = new Car("붕붕");
        v[1] = new Bicycle("따릉이");
        for(Vehicle s : v) {
            System.out.println(s);
            s.displayInfo();
            s.move();
        }
    }
}