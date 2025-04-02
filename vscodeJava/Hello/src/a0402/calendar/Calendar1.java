package a0402.calendar;

import java.util.Calendar;

public class Calendar1 {
    public static void main(String[] args) {
        Calendar calender = Calendar.getInstance();

        int year = calender.get(Calendar.YEAR);
        int month = calender.get(Calendar.MONTH)+1;
        int day = calender.get(Calendar.DAY_OF_MONTH);
        int hour = calender.get(Calendar.HOUR_OF_DAY);
        int minute = calender.get(Calendar.MINUTE);
        int second = calender.get(Calendar.SECOND);

        // 날짜 및 시간 출력
        System.out.println("현재 시간 : " + year + "-" + month + "-" + day + " " + hour + ":" + minute + ":" + second);
    }
}
