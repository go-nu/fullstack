package com.springboot.repository;

import com.springboot.domain.Book;

import java.util.List;

public interface BookRepository {
    List<Book> getAllBookList();
}
