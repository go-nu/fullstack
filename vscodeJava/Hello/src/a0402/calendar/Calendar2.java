package a0402.calendar;

import java.util.Calendar;

public class Calendar2 {
    public static void main(String[] args) {
        Calendar startDate = Calendar.getInstance();
        startDate.set(1999, Calendar.MARCH, 23);

        Calendar endDate = Calendar.getInstance();
        endDate.set(2025, Calendar.APRIL, 2);

        long startMillis = startDate.getTimeInMillis();
        // getTimeInMillis() 1970.1.1부터 지정한 시간까지 millisecond계산
        // System.out.println(startMillis);
        long endMillis = endDate.getTimeInMillis();

        long diff = endMillis - startMillis;
        long diffDays = diff / (60 * 60 * 24 * 1000);
        System.out.println(diffDays);
    }
}
