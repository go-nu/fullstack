package com.example.demo.entity;


// import com.example.demo.constant.ProductModelSelect;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.util.Objects;

@Entity
@Getter
@Setter
public class ProductModel {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // private ProductModelSelect productModelSelect; // 옵션
    private String productModelSelect; // 옵션명(자유입력)
    private Integer price;  // 옵션별 가격
    private Integer prStock; // 재고 수량

    private String imageUrl;

    @ManyToOne
    @JoinColumn(name = "product_id")
    private Product product;

    @Override
    public boolean equals(Object o) {
        // 자기 자신과 비교하는 경우 true (성능 최적화)
        if (this == o) return true;

        // 비교 대상이 null이거나 클래스 타입이 다르면 false
        if (!(o instanceof ProductModel)) return false;

        // 다운캐스팅하여 실제 값을 비교
        ProductModel that = (ProductModel) o;

        // 중요 속성 값들이 모두 같을 때만 같은 객체로 판단
        // 여기서는 모델명(productModelSelect), 가격(price), 재고(prStock)를 기준으로 비교
        return Objects.equals(productModelSelect, that.productModelSelect)
                && Objects.equals(price, that.price)
                && Objects.equals(prStock, that.prStock);
    }

    @Override
    public int hashCode() {
        // equals()에 사용된 필드들과 동일한 필드들로 해시코드 생성
        // 해시 기반 자료구조(Set, Map 등)에서 빠른 검색과 중복 방지를 위해 필요
        return Objects.hash(productModelSelect, price, prStock);
    }


}
