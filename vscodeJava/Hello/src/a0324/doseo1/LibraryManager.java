package a0324.doseo1;

import java.util.ArrayList;
import java.util.Scanner;

public class LibraryManager {
    private ArrayList<Library> librarys; // Library 객체를 리스트(묶음으로 저장할 수 있는 배열종류)
    private ArrayList<Library> bookLocation; // Library 객체 중 대여한 객체를 저장하는 리스트

    public LibraryManager() {
        librarys = new ArrayList<>(); // 전체 책을 담을 리스트
        bookLocation = new ArrayList<>(); // 대여한 책을 담을 리스트

        // 더미 데이터
        librarys.add(new Library("This is Java", "Shin", "SectionA", "979-11-691-229-8"));
        librarys.add(new Library("First Encounter with React", "Lee", "SectionB", "979-11-6921-169-7"));
        librarys.add(new Library("The Principles of Web Standards", "Ko", "SectionC", "979-11-6303-622-7"));
    }

    public void allLibrary() {
        System.out.println("대출 가능한 도서보기");
        for(int i = 0; i < librarys.size(); i++) {
            Library library = librarys.get(i);
            if(library.isAvaiable()) {
                System.out.println(library);
            }
        }
        // for(Library library : librarys) {
        //     if(library.isAvaiable() == true) {
        //         System.out.println(library);
        //     }
        // }
    }

    public boolean bookLocations(String name) {
        for(Library library : librarys) {
            if(library.getTitle().equalsIgnoreCase(name) && library.isAvaiable()) {// 모두 소문자로 바꿔서 비교
                library.book(); // 대출 불가능으로 전환
                bookLocation.add(library); // 대출한 책을 넣어놓는 list에 추가

                return true;
            } 
        }
        return false;
    }

    public void bookLocation() {
        System.out.println("대출한 도서 : ");
        for(Library location : bookLocation) {
            System.out.println(location);
        }
    }

    public void addLibrary(String newTitle, String newAuthor, String newLocation, String newISBN) {
        // Library abc = new Library(newTitle, newAuthor, newLocation, newISBN)
        // librarys.add(abc);
        librarys.add(new Library(newTitle, newAuthor, newLocation, newISBN));
    }

    public void delLibrary(String dname) {
        boolean result = false;
        for(Library library : librarys) {
            if (library.getTitle().equalsIgnoreCase(dname)) {
                if (library.isAvaiable()) {
                    librarys.remove(library);
                    result = true;
                    break;
                } else {
                    result = false;
                    break;
                }
            }
        }
        if(result) System.out.println("삭제됨");
        else System.out.println("삭제 안됨");
    }

    public void updateLibrary(String uname) {
        int i = 0;
        int index = -1;
        int menu = -1;
        boolean flag = true;
        Scanner sc = new Scanner(System.in);
        Library newA = new Library(); // 빈 라이브러리 객체 생성
        System.out.println(uname);
        for(Library a : librarys) { // librarys 리스트를 돌며 이름이 같은 객체 찾기
            i++;
            if(a.getTitle().equalsIgnoreCase(uname)) {
                index = i - 1;
                newA = a; // 이름이 같은 객체를 newA로 초기화
            }
            System.out.println(a.getTitle().equals(uname) + " " + a.getTitle() + " " + uname);
        }
        if (index != -1) { // index가 -1이 아니면 = 같은 이름의 객체를 찾으면
            System.out.print("뭘 수정할건데?\n 1.도서 이름 \t 2.도서 저자 \t 3.도서 위치 \t 4.도서ISBN \n >>");
            menu = sc.nextInt();
            sc.nextLine();
            while (flag) {
                switch (menu) {
                    case 1:
                        System.out.println("수정할 도서 이름 : ");
                        newA.setTitle(sc.nextLine()); // 키보드 입력으로 받은 문자열을 newA의 제목으로 변경
                        librarys.set(index, newA); // librarys(ArrayList)에 index자리에 newA를 초기화
                        flag = false;
                        break;
                    case 2:
                        System.out.println("수정할 도서 저자 : ");
                        newA.setAuthor(sc.nextLine());
                        librarys.set(index, newA);
                        flag = false;
                        break;
                    case 3:
                        System.out.println("수정할 도서 위치 : ");
                        newA.setLocation(sc.nextLine());
                        librarys.set(index, newA);
                        flag = false;
                        break;
                    case 4:
                        System.out.println("수정할 도서 ISBN : ");
                        newA.setIsbn(sc.nextLine());
                        librarys.set(index, newA);
                        flag = false;
                        break;
                    default:
                        System.out.println("올바른 번호를 입력하세요");
                        break;
                }
            }
        } else System.out.println("찾는 도서가 없습니다.");
    }

    public void showLibrary(String sname) {
        for(Library a : librarys) {
            if (a.getTitle().equalsIgnoreCase(sname)) {
                System.out.println(a.toString());
            }
        }
    }

}
