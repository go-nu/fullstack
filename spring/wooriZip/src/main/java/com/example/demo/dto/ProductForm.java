package com.example.demo.dto;


import com.example.demo.entity.*;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class ProductForm {
    private Long id;
    private String name;
//    private String description; // 설명
    private int price;
    private Category category;
    private int stockQuantity;

    private Long categoryId;

    // 상품 속성값 id 리스트 (예: 색상-화이트, 사이즈-퀸 등)
    private List<Long> attributeValueIds = new ArrayList<>();

    //  하위 모델들 (각 ProductModelDto는 여러 속성값 id를 가질 수 있음)
    private List<ProductModelDto> productModelDtoList = new ArrayList<>();
    private List<String> imageUrls;

    // 엔티티 변환
    public Product createProduct(Category category, Users user) {
        Product product = new Product();
        product.setName(this.name);
        product.setPrice(this.price);
        product.setCategory(category); // 엔티티 기준
        product.setUser(user);
        product.setAverageRating(0.0); // 별점 평균
        product.setStockQuantity(0);  // 초기 재고 수량 설정

        // 모델 정보 추가 (productModelDtoList)
        if (this.productModelDtoList != null && !this.productModelDtoList.isEmpty()) {
            for (ProductModelDto dto : this.productModelDtoList) {
                ProductModel model = new ProductModel();
                model.setProductModelSelect(dto.getProductModelSelect());    // 모델명(자유입력)
                model.setPrStock(dto.getPrStock());  // 재고 설정
                model.setPrice(dto.getPrice());  // 가격 설정
                model.setProduct(product);  // 상품에 모델 연결
                product.addProductModel(model);  // 상품에 모델 추가
                product.setStockQuantity(product.getStockQuantity() + dto.getPrStock());  // 재고 업데이트
            }
        }

        // ProductModelDto의 attributeValueIds는 서비스에서 ProductModelAttribute로 변환/연결 처리 예정
        return product;
    }


    public static ProductForm from(Product product) {
        ProductForm form = new ProductForm();
        form.setId(product.getId());
        form.setName(product.getName());
        form.setPrice(product.getPrice());

        // 모델 정보 매핑 (옵션)
        if (product.getProductModels() != null && !product.getProductModels().isEmpty()) {
            List<ProductModelDto> modelDtoList = product.getProductModels().stream()
                    .map(model -> {
                        ProductModelDto dto = new ProductModelDto();
                        dto.setProductModelSelect(model.getProductModelSelect());
                        dto.setPrStock(model.getPrStock());
                        dto.setImageUrl(model.getImageUrl());
                        return dto;
                    })
                    .toList();
            form.setProductModelDtoList(modelDtoList);
        }

        // ✅ 이미지 정보 매핑
        if (product.getImages() != null && !product.getImages().isEmpty()) {
            List<String> imageUrls = product.getImages().stream()
                    .map(ProductImage::getImageUrl)
                    .toList();
            form.setImageUrls(imageUrls);
        }

        return form;
    }

}
