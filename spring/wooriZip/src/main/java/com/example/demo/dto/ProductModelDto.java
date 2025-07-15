package com.example.demo.dto;


import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter @Setter
@NoArgsConstructor
public class ProductModelDto {
    private Long id;
    private String productModelSelect; // 옵션명(자유입력)
    private Integer price;                         // 옵션 가격
    private Integer prStock;                        // 기종별 재고 수량
    private String imageUrl;                        // ✅ 모델별 이미지 URL
    private String modelName; // 프론트에서 넘어오는 옵션명(임시)

    public void setModelName(String modelName) {
        this.modelName = modelName;
        // String -> Enum 변환 (예: "퀸" -> ProductModelSelect.QUEEN)
        // enum 삭제로 단순 대입
        this.productModelSelect = modelName;
    }
}
