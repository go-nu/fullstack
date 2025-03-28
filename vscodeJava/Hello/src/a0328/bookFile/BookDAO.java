package a0328.bookFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;


public class BookDAO {
    private ArrayList<BookDTO> books;
    Scanner s = new Scanner(System.in);
    FileClass file = new FileClass("library", "books");

    public BookDAO() {
        books = new ArrayList<BookDTO>();

        books.add(new BookDTO("자바의 정석", "남궁성", "12345", 32000));
        books.add(new BookDTO("Effective Java", "Joshua Bloch", "54321", 45000));
        books.add(new BookDTO("Clean Code", "Robert C. Martin", "67890", 38000));
        books.add(new BookDTO("스프링 부트와 AWS", "이동욱", "98765", 28000));
        books.add(new BookDTO("코틀린 인 액션", "Dmitry Jemerov", "13579", 40000));
    }

    private int searchISBN(String t) {
        int index = -1;
        for(int i = 0; i < books.size(); i++) {
            if(books.get(i).getISBN().equals(t)) {
                index = i;
                break;
            }
        }
        return index;
    }

	public void addBook() {
		System.out.print("추가할 도서의 제목 입력 : ");
        String aTitle = s.nextLine();
        System.out.print("추가할 도서의 저자 입력 : ");
        String aAuthor = s.nextLine();
        System.out.print("추가할 도서의 ISBN 입력 : ");
        String aISBN = s.nextLine();
        System.out.print("추가할 도서의 가격 입력 : ");
        int aPrice = s.nextInt();
        s.nextLine();

        books.add(new BookDTO(aTitle, aAuthor, aISBN, aPrice));
        System.out.println(aTitle + "이(가) 추가 되었습니다.");
	}

    public void deleteBook() {
        System.out.print("삭제할 도서의 ISBN 입력 : ");
        String dISBN = s.nextLine();
        int i = searchISBN(dISBN);
        if(i == -1) {
            System.out.println("찾는 ISBN의 도서가 없습니다.");
        } else {
            books.remove(books.get(i));
            System.out.println(books.get(i).getTitle() + "이(가) 삭제되었습니다.");
        }
    }

    public void searchBook() {
        System.out.println("검색할 도서의 ISBN 입력 : ");
        String sISBN = s.nextLine();
        int i = searchISBN(sISBN);
        if(i == -1) {
            System.out.println("찾는 ISBN의 도서가 없습니다.");
        } else {
            System.out.println(books.get(i).toString());
        }
    }

    public void updateBook() {
        System.out.println("수정할 도서의 ISBN 입력 : ");
        String uISBN = s.nextLine();
        int i = searchISBN(uISBN);
        if(i == -1) {
            System.out.println("찾는 ISBN의 도서가 없습니다.");
        } else {
            System.out.print("수정할 도서의 가격 입력 : ");
            int uPrice = s.nextInt();
            s.nextLine();
            books.get(i).setPrice(uPrice);
            System.out.println("도서의 가격이 수정되었습니다.");
        }

    }

    public void printAll() {
        for(int i = 0; i < books.size(); i++) {
            System.out.println(books.get(i));
        }
    }

    public void saveFile() throws IOException{
        file.create();
        String str = "";
        for(int i = 0; i < books.size(); i++) {
            str += books.get(i).toString()+"\n";
        }
        file.write(str);
    }

    public void loadFile() {
        try {
            file.read();    
        } catch (Exception e) {
            System.out.println("읽을 파일이 없습니다.");
        }
    }








}

