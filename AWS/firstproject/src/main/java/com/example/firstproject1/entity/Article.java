package com.example.firstproject1.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;

@AllArgsConstructor
@ToString
@Entity
@Getter
@NoArgsConstructor
public class Article {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column // 2.Column 어노테이션 추가
    private String title;
    @Column
    private String content;


    public void patch(Article article) {

        if(article.title != null) {
            this.title = article.title;
        }
        if(article.content != null) {
            this.content = article.content;
        }
    }
}
