package a0324.doseo1;

import java.util.ArrayList;

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

}
