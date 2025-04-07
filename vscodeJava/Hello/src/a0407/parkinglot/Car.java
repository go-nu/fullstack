package a0407.parkinglot;

public class Car {
    private String carNumber; // 번호판
    private int inTime; // 입차 시간

    public Car(String carNumber, int inTime) {
        this.carNumber = carNumber;
        this.inTime = inTime;
    }

    public String getCarNumber() {
        return carNumber;
    }

    public int getInTime() {
        return inTime;
    }

    @Override
    public String toString() {
        return "차량번호 : " + carNumber + ", 입차시간 : " + inTime;
    }
}