package com.shop.dto;

import com.querydsl.core.annotations.QueryProjection;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class MainItemDto {

    private Long id;
    private String itemNm;
    private String itemDetail;
    private String imgUrl;
    private Integer price;

    // QMainItemDto 자동 생성
    @QueryProjection
    public MainItemDto(Long id, String itemNm, String itemDetail, String imgUrl, int price){
        this.id = id;
        this.itemNm = itemNm;
        this.itemDetail = itemDetail;
        this.imgUrl = imgUrl;
        this.price = price;
    }


}
