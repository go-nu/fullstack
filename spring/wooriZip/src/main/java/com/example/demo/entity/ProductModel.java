package com.example.demo.entity;


import com.example.demo.constant.ProductModelSelect;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

@Entity
@Getter
@Setter
public class ProductModel {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    private ProductModelSelect productModelSelect; // 옵션
    private Integer price;  // 옵션별 가격
    private Integer prStock; // 재고 수량

    private String imageUrl;

    @ManyToOne
    @JoinColumn(name = "product_id")
    private Product product;


}
