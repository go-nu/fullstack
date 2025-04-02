package a0402.calendar;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class DateEx {
    public static void main(String[] args) {
        LocalDateTime now = LocalDateTime.now(); // 현재 날짜와 시간
        System.out.println("기본 ISO 형식 : " + now);
        // 커스텀 포맷
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd hh:mm:ss a");
        String formattedDate = now.format(formatter);
        System.out.println(formattedDate);
    }
}
// mm(소문자)는 분으로 인식
// MM DD 두개를 쓰면 04 02와 같이 0이 채워짐
// HH(대문자) = 24시간 표기, hh(소문자) = 12시간 표기
// hh:mm:ss a : a는 am, pm 표기