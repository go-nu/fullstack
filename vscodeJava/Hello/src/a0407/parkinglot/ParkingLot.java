package a0407.parkinglot;

import java.util.ArrayList;
import java.util.Scanner;

public class ParkingLot {
    Scanner sc = new Scanner(System.in);
    ArrayList<Car> cars = new ArrayList<>(5);
    
    public void parkCar(String number, int inTime) {
        boolean sameCar = false;
        for(int i = 0; i < cars.size(); i++) {
            if (number.equals(cars.get(i).getCarNumber())) {
                System.out.println("이미 등록된 차량입니다.");
                sameCar = true;
                break;
            }
        }
        if(!sameCar) {
            if (cars.size() > 5) {
                System.out.println("이 주차장이 자리가 가득 찼습니다.");
            } else {
                Car newCar = new Car(number, inTime);
                cars.add(newCar);
                System.out.println(newCar.getCarNumber() + " 입차 완료.");
            }
        }
    }

    public void exitCar(String number, int outTime) {
        int pay = 0;
        if (cars != null && !cars.isEmpty()) {
            for(int i = 0; i < cars.size(); i++) {
                if (number.equals(cars.get(i).getCarNumber())) {
                    pay = ((outTime - cars.get(i).getInTime())/10)*1000;
                    cars.remove(i);
                    break;
                }
            }
            System.out.println("요금 : " + pay + "원");
        } else {
            System.out.println("이 주차장은 현재 비어있습니다.");
        }
    }

    public void showCars() {
        if (cars != null && !cars.isEmpty()) {
            System.out.println("현재 주차 차량");
            for(Car c : cars) {
                System.out.println(c);
            }
        } else {
            System.out.println("이 주차장은 현재 비어있습니다.");
        }
    }

}
