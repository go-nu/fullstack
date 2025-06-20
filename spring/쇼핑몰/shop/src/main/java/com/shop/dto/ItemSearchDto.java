package com.shop.dto;

import com.shop.constant.ItemSellStatus;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ItemSearchDto {

    private String searchDateType; // 작성일

    private ItemSellStatus searchSellStatus; // 판매 상태

    private String searchBy; // 작성자

    private String searchQuery = ""; // 검색어
}
