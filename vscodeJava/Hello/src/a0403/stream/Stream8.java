package a0403.stream;

import java.util.ArrayList;
import java.util.List;

public class Stream8 {
    public static void main(String[] args) {
        List<Member> list = new ArrayList<>();
        Member m1 = new Member("박태호",Member.MALE,30);
        Member m2 = new Member("김연경",Member.FEMALE, 29);
        Member m3 = new Member("손유일", Member.MALE, 32);
        Member m4 = new Member("안재홍", Member.MALE, 27);
        list.add(m1);
        list.add(m2);
        list.add(m3);
        list.add(m4);
        
        int count  = 0;
        double sum = 0;
        double ageAvg1 = 0;
        // 외부 반복자
        for(int i = 0; i < list.size(); i++) {
            if(list.get(i).getGender() == Member.MALE) {
                sum += list.get(i).getAge();
                count++;
            }
        }
        ageAvg1 = sum / count;
        System.out.println("남성 나이 평균: " + ageAvg1);

        // 내부 반복자
        double ageAvg = list.stream() // list를 stream으로 변경
            .filter(m -> m.getGender() == Member.MALE) // 중간연산
            .mapToInt(Member::getAge) // 객체 형태를 int형으로 변환 (Member 객체를 나이로 매칭)
            .average() // 평균 계산, 내부 함수
            .getAsDouble(); // 평균값을 double로 변환
        System.out.println("남성 나이 평균: " + ageAvg);
    }
}
