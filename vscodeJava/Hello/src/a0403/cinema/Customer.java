package a0403.cinema;

import java.time.LocalDate;

public class Customer {
    private String name; // 이름
    private int birthYear; // 태어난 년도 (4자리 입력)
    private String pw; // 예약 비밀번호
    private String seat; // 예약 좌석
    
    // 예약 전 정보 저장을 위한 생성자
    public Customer(String name, int birthYear) {
        this.name = name;
        this.birthYear = birthYear;
    }
    // 예약 후 정보 저장을 위한 생성자
    public Customer(String name, int birthYear, String pw) {
        this.name = name;
        this.birthYear = birthYear;
        this.pw = pw;
    }

    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public int getBirthYear() {
        return birthYear;
    }
    public void setBirthYear(int birthYear) {
        this.birthYear = birthYear;
    }
    public String getReservationNum() {
        return pw;
    }
    public void setReservationNum(String pw) {
        this.pw = pw;
    }
    public String getSeat() {
        return seat;
    }
    public void setSeat(String seat) {
        this.seat = seat;
    }

    // 19세 이상임을 계산하는 메소드
    public boolean plus19(Customer c) {
        LocalDate today = LocalDate.now();
        int y = today.getYear();

        int age = y - birthYear;

        return age > 19;        
    }
}
