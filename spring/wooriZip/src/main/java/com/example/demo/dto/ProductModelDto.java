package com.example.demo.dto;


import com.example.demo.constant.ProductModelSelect;
import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class ProductModelDto {
    private Long id;
    private ProductModelSelect productModelSelect; // enum으로 "퀸", "슈퍼싱글" 구분
    private Integer price;                         // 옵션 가격
    private Integer prStock;                        // 기종별 재고 수량
    private String imageUrl;                        // ✅ 모델별 이미지 URL
}
