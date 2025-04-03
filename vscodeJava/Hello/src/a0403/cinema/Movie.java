package a0403.cinema;

import java.text.DecimalFormat;
import java.util.ArrayList;

public class Movie {
    private String title; // 영화 제목
    private String time; // 상영시간
    private int price; // 가격
    private boolean adult; // 청불
    private ArrayList<String> seats; // 좌석

    public Movie(String title, String time, int price, boolean adult) {
        this.title = title;
        this.time = time;
        this.price = price;
        this.adult = adult;
        seats = new ArrayList<>();
        for (int i = 65; i < 71; i++) { // A ~ F
            for (int j = 1; j < 7; j++) { // 1 ~ 6
                char asc = (char)i;
                String seat = asc + String.valueOf(j);
                seats.add(seat);
            }
        }
    }

    public String getTitle() {
        return title;
    }
    public void setTitle(String title) {
        this.title = title;
    }
    public String getTime() {
        return time;
    }
    public void setTime(String time) {
        this.time = time;
    }
    public int getPrice() {
        return price;
    }
    public void setPrice(int price) {
        this.price = price;
    }
    public boolean isAdult() {
        return adult;
    }
    public void setAdult(boolean adult) {
        this.adult = adult;
    }
    public ArrayList<String> getSeats() {
        return seats;
    }
    public void setSeats(ArrayList<String> seats) {
        this.seats = seats;
    }

    @Override
    public String toString() {
        return "[제목 : " + title + ", 시작시간 : " + time + ", 가격 : " + price + "]";
    }

    // 좌석 표시
    public void seatToString() {
        for (int i = 0; i < seats.size()-5; i +=6) {
            System.out.printf(" |  [%2s]\t[%2s][%2s]\t[%2s][%2s]\t[%2s]  |\n", seats.get(i), seats.get(i+1), seats.get(i+2), seats.get(i+3), seats.get(i+4), seats.get(i+5));
        }
    }

}
