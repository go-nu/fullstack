package a0324.yanolla;

import java.util.Scanner;

public class Search {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        Manager m = new Manager();
        boolean flag = true;

        while (flag) {
            System.out.println("\n숙소 예약 시스템에 오신 것을 환영합니다.");
            System.out.println("1. 예약 가능한 숙소 보기");
            System.out.println("2. 숙소 예약하기");
            System.out.println("3. 예약한 숙소 보기");
            System.out.println("4. 숙소 추가하기");
            System.out.println("5. 숙소 삭제하기");
            System.out.println("6. 숙소 정보 수정하기");
            System.out.println("7. 숙소 정보 조회하기");
            System.out.println("8. 종료");
            System.out.print("원하는 작업을 선택하세요 > ");

            int select = s.nextInt();
            s.nextLine();
            switch (select) {
                case 1:
                System.out.println("예약 가능한 숙소> ");
                    m.bookableList();
                    break;
                case 2:
                    System.out.println("숙소 예약하기> ");
                    System.out.print("예약할 호텔명을 입력하세요 : ");
                    String reserveAccommodationName = s.nextLine();
                    if(m.reserveAccommodation(reserveAccommodationName)) {
                        System.out.println("해당 숙소가 예약되었습니다.");
                    } else System.out.println("해당 숙소가 없거나 이미 예약되었습니다.");
                    break;
                case 3:
                    System.out.println("예약한 숙소> ");
                    m.checkReserved();
                    break;
                case 4:
                    System.out.println("숙소 추가하기> ");
                    System.out.println("추가할 숙소 명 : ");
                    String addName = s.nextLine();
                    System.out.println("추가할 숙소 위치 : ");
                    String addLocation = s.nextLine();
                    System.out.println("추가할 숙소 가격 : ");
                    double addPrice = s.nextDouble();
                    s.nextLine();
                    m.addAccommodation(addName, addLocation, addPrice);
                    System.out.println("숙소가 추가되었습니다.");
                    break;
                case 8:
                    flag = false;
                    break;
            
                default:
                    break;
            }




        }
        System.out.println("프로그램 종료");
        s.close();
    }
}
