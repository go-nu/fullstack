package com.springboot.repository;

import com.springboot.domain.Book;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

@Repository
public class BookRepositoryImpl implements BookRepository {

    private List<Book> listOfBooks = new ArrayList<Book>();

    public BookRepositoryImpl() {
        Book book1 = new Book();
        book1.setBookId("ISBN1234");
        book1.setName ("자바스크립트 입문");
        book1.setUnitPrice(new BigDecimal(30000));
        book1.setAuthor("조현영");
        book1.setDescription(
                "자바스크립트의 기초부터 심화까지 핵심 문법을 학습한 후 12가지 프로그램을 만들며 학습한 내용을 확인할 수 있습니다. 문법 학습과 실습이 적절히 섞여 있어 프로그램을 만드는 방법을 재미있게 익힐 수 있습니다.");
        book1.setPublisher("길벗");
        book1.setCategory("IT전문서");
        book1.setUnitsInStock(1000);
        book1.setReleaseDate("2024/02/20");

        Book book2 = new Book();
        book2.setBookId("ISBN1235");
        book2.setName ("파이썬의 정석");
        book2.setUnitPrice(new BigDecimal(29800));
        book2.setAuthor("조용주,임좌상");
        book2.setDescription(
                "4차 산업혁명의 핵심인 머신러닝, 사물 인터넷(IoT), 데이터 분석 등 다양한 분야에 활용되는 직관적이고 간결한 문법의 파이썬 프로그래밍 언어를 최신 트렌드에 맞게 예제 중심으로 학습할 수 있습니다.");
        book2.setPublisher("길벗");
        book2.setCategory("IT교육교재");
        book2.setUnitsInStock(1000);
        book2.setReleaseDate("2023/01/10");

        Book book3 = new Book();
        book3.setBookId("ISBN1236");
        book3.setName ("안드로이드 프로그래밍");
        book3.setUnitPrice(new BigDecimal(36000));
        book3.setAuthor("송미영");
        book3.setDescription(
                "안드로이드의 기본 개념을 체계적으로 익히고, 이를 실습 예제를 통해 익힙니다. 기본 개념과 사용법을 스스로 실전에 적용하는 방법을 학습한 다음 실습 예제와 응용 예제를 통해 실전 프로젝트 응용력을 키웁니다.");
        book3.setPublisher("길벗");
        book3.setCategory("IT교육교재");
        book3.setUnitsInStock(1000);
        book3.setReleaseDate("2023/06/30");

        listOfBooks.add(book1);
        listOfBooks.add(book2);
        listOfBooks.add(book3);
    }

    @Override
    public List<Book> getAllBookList() {
        return listOfBooks;
    }
}
