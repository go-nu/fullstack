package a0403.cinema;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class CinemaManager {
    private static ArrayList<Movie> movies; // 영화 목록을 저장하는 ArrayList
    private static ArrayList<Customer> customers; // 예매한 손님 목록을 저장하는 ArrayList

    // K : 손님 이름, V : 영화를 가지는 Map
    private static Map<String, Movie> reservationMap = new HashMap<>();

    private static FileC fc = new FileC();
    Scanner sc = new Scanner(System.in);

    public CinemaManager() {
        movies = new ArrayList<>();
        movies.add(new Movie("재밌는 영화", "11:00", 12000 ,false));
        movies.add(new Movie("슬픈 영화", "13:00", 12000 ,false));
        movies.add(new Movie("3D 영화", "14:00", 30000 ,false));
        movies.add(new Movie("잔인한 영화", "19:00", 12000 ,true));
        customers = new ArrayList<>();
        Movie tm = movies.get(0);
        reservationMap.put("테스트", tm);
    }

    public void displayMovies() {
        System.out.println("==================== 상영중인 영화 목록 ====================");
        int i = 1;
        for(Movie m : movies) {
            System.out.println("Movie" + i + ". " + m);
            i++;
        }
        System.out.println("============================================================");
    }

    public void bookTicket() {
        for(;;) {
            displayMovies();
            System.out.print("예매하실 영화의 번호를 입력해주세요 : ");
            try {
                int mNum = sc.nextInt();
                sc.nextLine();
                if (mNum < 1 || mNum > movies.size()) { // 선택한 번호가 목록에 없는 번호인 경우
                    System.out.println("잘못된 입력입니다.");
                    continue; // for문으로 복귀
                }
                System.out.println("예매한 영화 정보 >>>");
                System.out.println("Movie" + mNum + ". " + movies.get(mNum-1)); // 표기된 영화는 1부터 시작하지만 실제 index는 0부터 시작하므로 -1
                Movie sMovie = movies.get(mNum-1); // 선택한 영화
                if (sMovie.isAdult()) { // 성인이면
                    System.out.println("19세 이상 관람 가능합니다.");
                    customerInfo(sMovie);
                }
                else {
                    customerInfo(sMovie);
                }
                // 좌석 예매
                if (!customers.isEmpty()) {
                    // seatSelection이 반환하는 int형의 좌석 번호를 String형으로 변환
                    String sSeat = Integer.toString(seatSelection(sMovie));
                    int a = seatSelection(sMovie);
                    // 손님들 list의 마지막 손님(customerInfo()에서 add한 분)의 자리 지정
                    System.out.println(sSeat);
                    System.out.println(sMovie.getSeats().get(a));
                    customers.get(customers.size()-1).setSeat(sSeat);
                    reservationMap.put(customers.get(customers.size()-1).getName(), sMovie);
                    break;
                }



            } catch (Exception e) {
                System.out.println("잘못된 입력입니다.");
            }
        }
    }

    private int seatSelection(Movie sMovie) {
        int seatNum = -1; // 좌석 번호
        int s2n = -1; // String으로 입력된 좌석을 숫자로 바꿔줄 변수
        sMovie.seatToString(); // 영화관 좌석 출력
        System.out.print("원하는 좌석을 입력해주세요 : ");
        String sSeat = sc.nextLine();
        String firstWord = String.valueOf(sSeat.charAt(0));
        int secondWord = sSeat.charAt(1);
        // A : 1~6, B : 7~12, C : 13~18, D : 19~24, E : 25~30, F : 31~36
        // charAt(1)로 마지막 글자를 뽑아낸 뒤 int 형으로 바꾸면 asc가 나옴
        // ex) "1" -> 49, "2" ->50 ...
        // A는 seats index 0부터 시작, B는 6부터 ...
        if (firstWord.equalsIgnoreCase("A")) {
            return s2n = secondWord-49; 
        } else if  (firstWord.equalsIgnoreCase("B")) {
            return s2n = secondWord-43; 
        } else if  (firstWord.equalsIgnoreCase("C")) {
            return s2n = secondWord-37; 
        } else if  (firstWord.equalsIgnoreCase("D")) {
            return s2n = secondWord-31; 
        } else if  (firstWord.equalsIgnoreCase("E")) {
            return s2n = secondWord-25; 
        } else if  (firstWord.equalsIgnoreCase("F")) {
            return s2n = secondWord-19;
        }        
        if(s2n+1 < 1 || s2n+1 > 36) {
            System.out.println("존재하지 않는 좌석입니다.");
        } else if (sMovie.getSeats().get(s2n).equals("XX")) {
            System.out.println("이미 선택된 좌석입니다.");
        } else {
            sMovie.getSeats().set(s2n, "XX");
            System.out.println("좌석 선택이 완료되었습니다.");
            seatNum = s2n;
        }
        
        
        return seatNum;
    }

    private void customerInfo(Movie sMovie) {
        System.out.print("예매자 성함을 입력해주세요 : ");
        String cName = sc.nextLine();
        System.out.print("예약자 생년(4자리)를 입력해주세요 : ");
        int cBirthYear = sc.nextInt();
        sc.nextLine();
        Customer c = new Customer(cName, cBirthYear);
        if (sMovie.isAdult() && !c.plus19(c)) {
            System.out.println("해당 영화는 청소년 관람이 불가합니다.");
        } else {
            System.out.print("예약 비밀번호를 입력해주세요 : ");
            String cPw = sc.nextLine();
            c = new Customer(cName, cBirthYear, cPw);
            customers.add(c);
            System.out.println("예약이 완료되었습니다.");
        }
    }
    
}
