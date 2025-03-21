package a0321.Library;

import java.util.Scanner;

public class LibraryApplication {
    private static Book[] bookArray = new Book[100];
    private static Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        boolean run = true;

        while(run) {
            System.out.println("-------------------------------------------------------------");
            System.out.println("1. 책 추가 | 2. 책 목록 조회 | 3. 책 대출 | 4. 책 반납 | 5. 종료");
            System.out.println("-------------------------------------------------------------");
            System.out.print("선택> ");
            int select = Integer.parseInt(scanner.nextLine());

            if (select == 1) {
                addBook();
            } else if (select == 2) {
                listBook();
            } else if (select == 3) {
                borrowBook();
            } else if (select == 4) {
                returnBook();
            } else if (select == 5) {
                run = false;
            }
        }
        System.out.println("프로그램 종료");
    }

    private static void addBook(){
        System.out.print("책 제목: ");
        String t = scanner.nextLine();
        System.out.print("저자: ");
        String a = scanner.nextLine();
        String s = "Available"; // 있으면 Available(대출 가능), 없으면 Borrowed(대출 불가)

        Book newBook = new Book(t, a, s);
        for(int i = 0; i < bookArray.length; i++) {
            if(bookArray[i] == null) {
                bookArray[i] = newBook;
                System.out.println("책이 추가되었습니다!");
                break;
            }
        }
    }

    private static void listBook(){
        for(int i = 0; i < bookArray.length; i++) {
            Book books = bookArray[i];
            if(bookArray[i] != null) {
                System.out.printf("책 제목 : %s | 저자 : %s | 상태 : %s\n", books.getTitle(), books.getAuthor(), books.getStatus());
            }
        }
    }

    private static void borrowBook(){
        System.out.print("대출할 책 제목: ");
        String t = scanner.nextLine();
        Book bBook = findBook(t);
        if(bBook != null) {
            if(bBook.getStatus() == "Available") {
                System.out.println("책을 대출했습니다!");
                bBook.setStatus("Borrowed");
            } else System.out.println("이미 대출중입니다.");
        } else System.out.println("없는 책입니다.");
    }

    private static void returnBook(){
        System.out.print("반납할 책 제목: ");
        String t = scanner.nextLine();
        Book bBook = findBook(t);
        if(bBook != null) {
            if(bBook.getStatus() == "Borrowed") {
                System.out.println("책을 반납했습니다!");
                bBook.setStatus("Available");
            } else System.out.println("반납된 책입니다.");
        } else System.out.println("없는 책입니다.");
    }

    private static Book findBook(String s) {
        Book fbook = null;
        for(int i = 0; i < bookArray.length; i++) {
            if(bookArray[i] != null) {
                String fbTitle = bookArray[i].getTitle();
                if(fbTitle.equals(s)) {
                    fbook = bookArray[i];
                    break;
                }
            }
        }
        return fbook;
    }
    
}
