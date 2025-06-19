package com.shop.dto;

import com.shop.entity.ItemImg;
import lombok.Getter;
import lombok.Setter;
import org.modelmapper.ModelMapper;

@Getter
@Setter
public class ItemImgDto {

    private Long id;

    private String imgName;

    private String oriImgName;

    private String imgUrl;

    private String repImgYn;

    /*public static ItemImgDto fromEntity(ItemImg entity) {
        ItemImgDto dto = new ItemImgDto();
        dto.setId(entity.getId());
        dto.setImgName(entity.getImgName());
        dto.setOriImgName(entity.getOriImgName());
        dto.setImgUrl(entity.getImgUrl());
        dto.setRepImgYn(entity.getRepImgYn());
        return dto;
    }*/

    // ENTITY를 DTO로 변환
    private static ModelMapper modelMapper = new ModelMapper();

    public static ItemImgDto of(ItemImg itemImg) {
        return modelMapper.map(itemImg,ItemImgDto.class);
    }
}
