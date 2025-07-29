package com.example.demo.entity;

import jakarta.persistence.*;

import java.time.LocalDateTime;

@Entity
public class Review { // 후기(리뷰)
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String content;
    private int rating; // 점수

    @ManyToOne
    private Users user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "product_id") // 0727 DK가 추가
    private Product product;

    private LocalDateTime createdAt;
}
