package com.shop.dto;

import com.shop.constant.ItemSellStatus;
import com.shop.entity.Item;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;
import org.modelmapper.ModelMapper;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class ItemFormDto {

    private Long id;

    @NotBlank(message = "상품명은 필수 입력 값입니다.")
    private String itemNm;

    @NotNull(message = "가격은 필수 입력 값입니다.")
    private Integer price;

    @NotBlank(message = "상품 상세는 필수 입력 값입니다.")
    private String itemDetail;

    @NotNull(message = "재고는 필수 입력 값입니다.")
    private Integer stockNumber;

    private ItemSellStatus itemSellStatus;

    private List<ItemImgDto> itemImgDtoList = new ArrayList<>();

    private List<Long> itemImgIds = new ArrayList<>();

    private static ModelMapper modelMapper = new ModelMapper();

    // dto -> entity
    public Item createItem(){
        return modelMapper.map(this, Item.class);
    }

    // entity -> dto
    public static ItemFormDto of(Item item){
        return modelMapper.map(item,ItemFormDto.class);
    }

    // 위의 내용을 builder로 바꾸면
    /*@Builder
    public ItemFormDto(Long id, String name, int price, String detail, int stockQuantity) {
        this.id = id;
        this.name = name;
        this.price = price;
        this.detail = detail;
        this.stockQuantity = stockQuantity;
    }

    // DTO → Entity 변환 (Builder 사용)
    public Item createItem() {
        return Item.builder()
                .id(this.id)
                .name(this.name)
                .price(this.price)
                .detail(this.detail)
                .stockQuantity(this.stockQuantity)
                .build();
    }

    // Entity → DTO 변환 (Builder 사용)
    public static ItemFormDto of(Item item) {
        return ItemFormDto.builder()
                .id(item.getId())
                .name(item.getName())
                .price(item.getPrice())
                .detail(item.getDetail())
                .stockQuantity(item.getStockQuantity())
                .build();
    }*/
}
