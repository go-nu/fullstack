package a0403.cinema;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;


public class CinemaManager {
    private static ArrayList<Movie> movies =new ArrayList<>(); // 영화 목록을 저장하는 ArrayList
    private static ArrayList<Customer> customers; // 예매한 손님 목록을 저장하는 ArrayList
    
    Customer admin = new Customer("Admin", 1980, "*123"); // 관리자 계정

    // K : 손님 이름, V : 영화를 가지는 Map
    private static Map<String, Movie> reservationMap = new HashMap<>();

    private static FileC fc = new FileC();  
    Scanner sc = new Scanner(System.in);

    public CinemaManager() {
        movies = new ArrayList<>();
        // fc.defaultMovie();
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
                if (sMovie.isAdult()) { // 청불 영화는 
                    System.out.println("19세 이상 관람 가능합니다.");
                    checkAge(sMovie);
                }
                else {
                    checkAge(sMovie);
                }
                // 좌석 예매
                if (customers != null && !customers.isEmpty()) {
                    // seatSelection이 반환하는 int형의 좌석 index번호를 String형으로 변환
                    String sSeat = Integer.toString(seatSelection(sMovie));
                    // 손님들 list의 마지막 손님(checkAge()에서 add한 분)의 자리 지정
                    customers.get(customers.size()-1).setSeat(sSeat);
                    System.out.println("예약중입니다.");
                    Thread.sleep(2000);
                    System.out.println("==============================================================");
                    System.out.println(customers.get(customers.size()-1).getName() + "님의 예약정보");
                    System.out.println("Movie" + mNum + ". " + sMovie + "\n좌석 : " + showSeat(sSeat));
                    System.out.println("==============================================================");
                    reservationMap.put(customers.get(customers.size()-1).getName(), sMovie);
                    break;
                }
            } catch (Exception e) {
                System.out.println("잘못된 입력입니다.");
            }
        }
    }

    public static String showSeat(String k) {
        int a = Integer.parseInt(k);
        String s = "";
    
        switch (a / 6) {
            case 0:
                s = "A" + ((a % 6) + 1);
                break;
            case 1:
                s = "B" + ((a % 6) + 1);
                break;
            case 2:
                s = "C" + ((a % 6) + 1);
                break;
            case 3:
                s = "D" + ((a % 6) + 1);
                break;
            case 4:
                s = "E" + ((a % 6) + 1);
                break;
            case 5:
                s = "F" + ((a % 6) + 1);
                break;
            default:
                break;
        }
    
        return s;
    }

    private int seatSelection(Movie sMovie) {
        int seatNum = -1; // 좌석 번호
        int s2n = -1; // String으로 입력된 좌석을 숫자로 바꿔줄 변수
        while (true) {
            try {
                sMovie.seatToString(); // 영화관 좌석 출력
                System.out.print("원하는 좌석을 입력해주세요 : ");
                String sSeat = sc.nextLine();
                String firstWord = String.valueOf(sSeat.charAt(0));
                int secondWord = sSeat.charAt(1);
                if (firstWord.equalsIgnoreCase("A")) {
                    s2n = secondWord-49;
                } else if  (firstWord.equalsIgnoreCase("B")) {
                    s2n = secondWord-43;
                } else if  (firstWord.equalsIgnoreCase("C")) {
                    s2n = secondWord-37;
                } else if  (firstWord.equalsIgnoreCase("D")) {
                    s2n = secondWord-31;
                } else if  (firstWord.equalsIgnoreCase("E")) {
                    s2n = secondWord-25;
                } else if  (firstWord.equalsIgnoreCase("F")) {
                    s2n = secondWord-19;
                }        
                if(s2n+1 < 1 || s2n+1 > 36) {
                    System.out.println("존재하지 않는 좌석입니다.");
                } else if (sMovie.getSeats().get(s2n).equals("XX")) {
                    System.out.println("이미 선택된 좌석입니다.");
                } else {
                    sMovie.getSeats().set(s2n, "XX");
                    System.out.println("좌석 선택이 완료되었습니다.");
                    seatNum = s2n;
                    break;
                }
            } catch (Exception e) {
                System.out.println("잘못된 입력입니다.");
                sc.nextLine();
            }
            
        }
        
        return seatNum;
    }

    private void checkAge(Movie sMovie) {
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
        }
    }

    public void displayTicket() {
        int index = -1;
        System.out.println("==================== 예매표 확인 ====================");
        System.out.print("예매자 성함을 입력해주세요 : ");
        String mName = sc.nextLine();
        for(int i = 0; i < customers.size(); i++) {
            if (mName.equals(customers.get(i).getName())) {
                index = i;
                break;
            }
        }
        if(index == -1) {
            System.out.println("예매자를 찾을 수 없습니다. 다시 입력해주세요");
            return;
        }
        System.out.print("비밀번호를 입력해주세요 : ");
        String mPw = sc.nextLine();
        if (mPw.equals(customers.get(index).getPw())) {
            System.out.println("비밀번호가 일치합니다.");
            if (customers != null) {
                for(int j = 0; j < customers.size(); j++) {
                    if(mName.equals(customers.get(j).getName())) {
                        String k = customers.get(j).getSeat();
                        String s = showSeat(k);
                        System.out.println("==============================================================");
                        System.out.println(customers.get(customers.size()-1).getName() + "님의 예약정보");
                        System.out.println("Movie" + (j+1) + ". " + reservationMap.get(mName) + "\n좌석 : " + s);
                        System.out.println("==============================================================");
                        break;
                    }
                }
            }
        }
        
        
    }

    public void cancleTicket() {
        int index = -1;
        System.out.println("==================== 예매 취소 ====================");
        if(customers == null || customers.isEmpty()) {
            System.out.println("예매자가 없읍니다. 예매를 먼저 진행해주세요.");
        }
        System.out.print("예매자 성함을 입력해주세요 : ");
        String mName = sc.nextLine();
        for(int i = 0; i < customers.size(); i++) {
            if (mName.equals(customers.get(i).getName())) {
                index = i;
                break;
            }
        }
        if(index == -1) {
            System.out.println("예매자를 찾을 수 없습니다. 다시 입력해주세요");
            return;
        }
        System.out.print("비밀번호를 입력해주세요 : ");
        String mPw = sc.nextLine();
        if (mPw.equals(customers.get(index).getPw())) {
            System.out.println("비밀번호가 일치합니다.");
            if (customers != null) {
                for(int j = 0; j < customers.size(); j++) {
                    if(mName.equals(customers.get(j).getName())) {
                        customers.remove(j);
                        break;
                    }
                }
            }
        }
        System.out.println("예매가 취소되었습니다. ");
    }

    public void checkCustomers() {
        // ArrayList<Movie> arr = new ArrayList<>();
        ArrayList<String> arr = new ArrayList<>();
        for(;;) {
            System.out.println("==================== 예매 고객 확인 ====================");
            if(ruAdmin()) {
                displayMovies();
                System.out.print("고객 정보를 확인할 영화를 선택해주세요 : ");
                int m = sc.nextInt()-1;
                sc.nextLine();
                System.out.println(movies.get(m) + "을 예매한 고객 정보 >>");
                for(int i = 0; i < movies.size(); i++) {
                    for(int j = 0; j < customers.size(); j++) {
                        if(movies.get(i).getTitle().equals(reservationMap.get(customers.get(j).getName()).getTitle())) {
                            // arr.add(reservationMap.get(customers.get(j).getName()));
                            arr.add(customers.get(j).getName() + "(" + customers.get(j).getPw() + ")");
                        }
                    }
                }
                System.out.println(arr);
                break;
            }
        }
        
    }

    private boolean ruAdmin() {
        System.out.print("관리자 이름 입력 : ");
            String aName = sc.nextLine();
            if (aName.equals(admin.getName())) {
                System.out.print("관리자 비밀번호 입력 : ");
                String aPw = sc.nextLine();
                if (aPw.equals(admin.getPw())) {
                    return true;
                }
            } else {
                System.out.println("잘못된 입력이거나 없는 관리자입니다.");
            } return false;
    }

    public void printTicket() {
        int index = -1;
        System.out.println("==================== 티켓 출력 ====================");
        if(customers == null || customers.isEmpty()) {
            System.out.println("예매자가 없읍니다. 예매를 먼저 진행해주세요.");
        }
        System.out.print("예매자 성함을 입력해주세요 : ");
        String mName = sc.nextLine();
        for(int i = 0; i < customers.size(); i++) {
            if (mName.equals(customers.get(i).getName())) {
                index = i;
                break;
            }
        }
        if(index == -1) {
            System.out.println("예매자를 찾을 수 없습니다. 다시 입력해주세요");
            return;
        }
        System.out.print("비밀번호를 입력해주세요 : ");
        String mPw = sc.nextLine();
        if (mPw.equals(customers.get(index).getPw())) {
            System.out.println("비밀번호가 일치합니다.");
            System.out.println("티켓이 출력됩니다.");
            fc.ticket2File(reservationMap, customers.get(index).getName());
        }
    }

    String ticket(Map<String, Movie> reservationMap, String name) {
        String str ="";
        if (customers != null) {
            for(int j = 0; j < customers.size(); j++) {
                if(name.equals(customers.get(j).getName())) {
                    String k = customers.get(j).getSeat();
                    String s = showSeat(k);
                    str = "==============================================================\n" + 
                    customers.get(customers.size()-1).getName() + "님의 예약정보\n" + 
                    "Movie" + (j+1) + ". " + reservationMap.get(name) + "\n좌석 : " + s + 
                    "\n==============================================================";
                }
            }
        }
        return str;
    }

    public static ArrayList<Movie> getMovies() {
        return movies;
    }

    public void addMovies() {
        if(ruAdmin()) {
            fc.addMovie();
        }
    }

    public void delMovies() {
        displayMovies();
        if(ruAdmin()) {
            System.out.print("삭제할 영화의 번호를 입력해주세요 : ");
            int delM = sc.nextInt()-1;
            sc.nextLine();
            if (delM < 0 || delM > movies.size()) {
                System.out.println("없는 영화 번호입니다.");
            } else {
                movies.remove(delM);
                System.out.println("영화가 삭제되었습니다.");
            }
        }
    }

    public void sortMovie() {
        for(int i = 0; i < movies.size(); i++) {
            for(int j = i + 1; j < movies.size(); j++) {
                int h1 = Integer.parseInt(movies.get(i).getTime().split(":")[0]);
                int h2 = Integer.parseInt(movies.get(j).getTime().split(":")[0]);
                int m1 = Integer.parseInt(movies.get(i).getTime().split(":")[1]);
                int m2 = Integer.parseInt(movies.get(i).getTime().split(":")[1]);

                if (h1 > h2) {
                    Movie temp = movies.get(i);
                    movies.set(i, movies.get(j));
                    movies.set(j, temp);
                } else if (h1 == h2) {
                    if (m1 > m2) {
                        Movie temp = movies.get(i);
                        movies.set(i, movies.get(j));
                        movies.set(j, temp);
                    }
                }
            }
        }
    }
}
