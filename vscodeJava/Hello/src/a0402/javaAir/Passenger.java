package a0402.javaAir;

import java.time.LocalDate;
import java.time.Period;

public class Passenger { // 예매 정보 저장 - 나이 검증
    private String name;
    private int birthDate; // 생년월일
    private String pw; // 예약 비밀번호
    private String seat; // 좌석

    public Passenger(String name, int birthDate) {
        this.name = name;
        this.birthDate = birthDate;
    }

    public Passenger(String name, int birthDate, String pw) {
        this.name = name;
        this.birthDate = birthDate;
        this.pw = pw;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getBirthDate() {
        return birthDate;
    }

    public void setBirthDate(int birthDate) {
        this.birthDate = birthDate;
    }

    public String getPw() {
        return pw;
    }

    public void setPw(String pw) {
        this.pw = pw;
    }

    public String getSeat() {
        return seat;
    }

    public void setSeat(String seat) {
        this.seat = seat;
    }
    
    // 15세 이상만 국제선 이용가능
    public boolean man15(Passenger p) {
        int y = p.birthDate / 10000;
        int m = (p.birthDate % 10000) / 100;
        int d = p.birthDate % 100;

        if(y > 0 && y < 25) {
            y += 2000;
        } else {
            y += 1900;
        }

        LocalDate birthDate2 = LocalDate.of(y, m, d); // 생년월일
        LocalDate currentDate = LocalDate.now(); // 오늘
        int age = Period.between(birthDate2, currentDate).getYears();

        return age > 15;
    }
    
}
