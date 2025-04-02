package a0402.calendar;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class DateEx1 {
    public static void main(String[] args) {
        // 1. 현재 날짜 및 시간 가져오기
        LocalDateTime now = LocalDateTime.now();
        System.out.println("현재 날짜와 시간: " + now);

        // 2. 특정 날짜 및 시간 설정
        LocalDateTime myDate = LocalDateTime.of(2025, 4, 2, 15, 30, 45);
        System.out.println("지정한 날짜와 시간: " + myDate);

        // 3. 날짜 및 시간 정보 가져오기
        System.out.println("연도: " + myDate.getYear());
        System.out.println("월: " + myDate.getMonthValue());
        System.out.println("일: " + myDate.getDayOfMonth());
        System.out.println("요일: " + myDate.getDayOfWeek());
        System.out.println("시간: " + myDate.getHour());
        System.out.println("분: " + myDate.getMinute());

        // 4. 날짜 계산
        LocalDateTime futureDate = myDate.plusDays(10);
        System.out.println("10일 후: " + futureDate);

        // 5. 날짜 비교
        LocalDateTime anotherDate = LocalDateTime.of(2026, 1, 1, 12, 0);
        System.out.println("미래 날짜인가? " + myDate.isBefore(anotherDate));

        // 6. 날짜 포맷 변환
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
        String formattedDate = myDate.format(formatter);
        System.out.println("포맷팅된 날짜: " + formattedDate);
    }
}
