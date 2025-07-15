package com.example.demo.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.LocalDate;
import java.util.List;

@Getter
@Builder
@ToString
public class OrderDto {

    // 주문 정보 Dto
    // 주문을 할 아이템들
    // 가격,개수,상품명=>찾기
    private Long orderNo; // 주문 번호
    private String orderId; // 주문 아이디 (uuid로 생성된 )
    private List<OrderItemDto> items;
    private int totalPrice;    // 총  결제 금액
    private int count; // 개수

    // form 에서 받을 데이터
    // 주문 고객 정보를 담을것
    private String userName; // 주문한 고객이름

    private String email; // 이메일

    private String phone; // 전화번호
    // 우편번호 기본주소 나머지주소
    private String pCode; // 우편번호
    private String loadAddress; // 도로
    private String lotAddress; //  지번
    private String detailAddress; // 상세

    private LocalDate orderTime;

    // 결제 성공시 넘길 내용
    private String payInfo; // 결제 수단
}
