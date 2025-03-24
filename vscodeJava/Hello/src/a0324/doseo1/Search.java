package a0324.doseo1;

import java.util.Scanner;

public class Search {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        LibraryManager manager = new LibraryManager();
        // 생성과 동시에 생성 메서드에 더미 데이터가 들어있어 만들어짐
        boolean flag = true;

        while(flag) {
            System.out.println("\n도서검색 시스템에 오신 것을 환영합니다.");
            System.out.println("1.대출 가능한 도서 보기");
            System.out.println("2.도서 대출하기");
            System.out.println("3.대출한 도서 보기");
            System.out.println("4.도서 추가하기");
            System.out.println("5.도서 삭제하기");
            System.out.println("6.도서 정보 수정하기");
            System.out.println("7.도서 내용 보기");
            System.out.println("8.종료");
            System.out.print("원하는 작업을 선택하세요 >> ");

            int choice = sc.nextInt();
            sc.nextLine(); // 이상한 문자 제거
            switch (choice) {
                case 1:
                    System.out.println("대출 가능한 도서");
                    manager.allLibrary(); // 대출 가능한 도서 조회 메소드
                    break;
                case 2:
                    System.out.println("도서 대출하기");
                    System.out.println("대출하려는 도서의 이름을 입력하세요 : ");
                    String libraryName = sc.nextLine();
                    if(manager.bookLocations(libraryName)) {
                        System.out.println("도서가 성공적으로 대출되었습니다.");
                    } else {
                        System.out.println("도서가 대출 가능하지 않거나, 존재하지 않습니다.");
                    }
                    break;
                case 3:
                    System.out.println("대출한 도서 보기");
                    manager.bookLocation();
                    break;
                case 4:
                    System.out.println("도서 추가하기");
                    System.out.print("추가 도서 이름 : ");
                    String newTitle = sc.nextLine();
                    System.out.print("추가 도서 저자 : ");
                    String newAuthor = sc.nextLine();
                    System.out.print("도서 위치 : ");
                    String newLocation = sc.nextLine();
                    System.out.print("도서 ISBN : ");
                    String newISBN = sc.nextLine();
                    sc.nextLine(); // 개행문자 초기화
                    manager.addLibrary(newTitle, newAuthor, newLocation, newISBN);
                    System.out.println("도서 추가 완료");
                    break;  
                case 8:
                    flag = false;
                    break;
                default:
                    break;
            }
        }







        sc.close();
    }
}
