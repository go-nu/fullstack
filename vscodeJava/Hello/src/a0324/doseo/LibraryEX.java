package a0324.doseo;

public class LibraryEX {
    public static void main(String[] args) {
        // EBook 객체 생성
        EBook eBook = new EBook("이것이 자바다", "신용권", "2023-1113", 5.6, "PDF");
        
        PrintBook printBook = new PrintBook("첨만나리액트", "이인제", "2019-0124", 464, 0.8);

        // 도서정보 출력
        System.out.println("EBook Information: ");
        eBook.displayInfo();
        System.out.println();
        System.out.println("Print Book information");
        printBook.displayInfo();
        
    }
}
