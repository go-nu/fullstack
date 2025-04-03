package a0403.cinema;

import java.util.Scanner;

public class CinemaMain {
    public static void main(String[] args) {
        CinemaManager cm = new CinemaManager();
        Scanner sc = new Scanner(System.in);
        FileC fc = new FileC();

        System.out.println("==================== 영화표 예매 프로그램 ====================");

        Outter:while (true) {
            System.out.println("1. 현재 상영중인 영화 목록\n2. 영화표 예매\n3. 예매한 영화표 조회\n4. 영화표 출력\n5. 종료");
            System.out.println("==============================================================");
            System.out.print("선택> ");

            String select = sc.next();
            sc.nextLine();

            int menu = -1;
            try {
                menu = Integer.parseInt(select);
            } catch (Exception e) {
                System.out.println("잘못된 입력입니다.");
                System.out.println("처음 화면으로 돌아갑니다.");
            }

            switch (menu) {
                case 1:
                    cm.displayMovies();
                    break;
                case 2:
                    cm.bookTicket();
                    break;
                case 3:
                    
                    break;
                case 4:
                    
                    break;
                case 5:
                    System.out.println("프로그램을 종료합니다.");
                    sc.close();
                    break Outter;
                default:
                    break;
            }
        }
    }
}
