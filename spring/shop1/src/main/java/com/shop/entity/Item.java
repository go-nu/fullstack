package com.shop.entity;

import com.shop.constant.ItemSellStatus;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import java.time.LocalDateTime;

@Entity
@Table(name="item")
@Getter
@Setter
@ToString
public class Item {

    @Id
    @Column(name="item_id")
    @GeneratedValue(strategy=GenerationType.IDENTITY)
    private Long id;  //상품 코드

    @Column(nullable=false, length = 50)
    private String itemNm;  //상품명

    @Column(name = "price", nullable = false)
    private int price;

    @Column(nullable=false)
    private int stockNumber;  //재고수량

    @Lob  // 글씨가 많을 경우 저장공간을 확보하는 어노테이션
    @Column(nullable=false)
    private String itemDetail; //상품 상세 설명


    private ItemSellStatus itemSellStatus; //상품 판매 상태
    private LocalDateTime regTime; //등록 시간
    private LocalDateTime updateTime; //수정 시간
}
