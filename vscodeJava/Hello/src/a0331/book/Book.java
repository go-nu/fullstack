package a0331.book;

import java.util.ArrayList;
import java.util.Map;
import java.util.LinkedHashMap;

public class Book {
    private static Book instance;
    private Book() {}
    public static Book getInstance() {
        if(instance == null) {
            instance = new Book();
        }
        return instance;
    }
    
    ArrayList<String> bookList;  // 책 목록
    ArrayList<Integer> bookPrice; // 책 가격 목록
    Map<String, Integer> menu;  // 책 메뉴

    public void getMenu() {
        menu = new LinkedHashMap<>();
        bookList = new ArrayList<>();
        bookPrice = new ArrayList<>();

        bookList.add("자바 프로그래밍");
        bookList.add("알고리즘");
        bookList.add("Spring 입문");
        bookList.add("JavaScript 완벽 가이드");
        bookList.add("데이터베이스 관리");

        bookPrice.add(15000);  // 자바 프로그래밍
        bookPrice.add(20000);  // 알고리즘
        bookPrice.add(18000);  // Spring 입문
        bookPrice.add(25000);  // JavaScript 완벽 가이드
        bookPrice.add(22000);  // 데이터베이스 관리

        for (int i = 0; i < bookList.size(); i++) {
            menu.put(bookList.get(i), bookPrice.get(i));
        }

        System.out.println("\n\n+----------------------------------------+");
        System.out.println("|                서점 메뉴판               |");
        System.out.println("+----------------------------------------+");
        int s = 1;
        for (Map.Entry<String, Integer> entry : menu.entrySet()) {
            System.out.printf("| [%d] %-20s %-10d원 |\n", s, entry.getKey(), entry.getValue());
            s++;
        }
        System.out.println("+----------------------------------------+");
    }
}
