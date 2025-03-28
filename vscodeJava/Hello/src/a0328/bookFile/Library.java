package a0328.bookFile;

import java.util.Scanner;

public class Library {
    public static void main(String[] args) {
        BookDAO m = new BookDAO();
        Scanner s = new Scanner(System.in);

        while(true) {
            System.out.println("===== 도서 관리 시스템 =====");
            System.out.println("1. 도서 추가");
            System.out.println("2. 도서 삭제");
            System.out.println("3. 도서 검색");
            System.out.println("4. 도서 수정");
            System.out.println("5. 도서 목록 보기");
            System.out.println("6. 파일로 저장");
            System.out.println("7. 파일에서 불러오기");
            System.out.println("0. 종료");
            System.out.print(">>");

            int num;
            try {
                num = s.nextInt();
            } catch (Exception e) {
                num = -1;
            }

            switch (num) {
                case 1:
                    m.addBook();
                    break;
                case 2:
                    m.deleteBook();
                    break;
                case 3:
                    m.searchBook();
                    break;
                case 4:
                    m.updateBook();
                    break;
                case 5:
                    m.printAll();
                    break;
                case 6:
                    try {
                        m.saveFile();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    break;
                case 7:
                    m.loadFile();
                    break;
                case 0:
                    System.out.println("프로그램 종료");
                    s.close();
                    System.exit(0);
                    break;
                default:
                    System.out.println("잘못된 입력입니다.");
                    break;
            }
        }

    }
}
