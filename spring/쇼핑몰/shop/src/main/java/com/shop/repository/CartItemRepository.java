package com.shop.repository;

import com.shop.dto.CartDetailDto;
import com.shop.entity.CartItem;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface CartItemRepository extends JpaRepository<CartItem, Long> {

    CartItem findByCartIdAndItemId(Long cartId, Long itemId);

    @Query("select new com.shop.dto.CartDetailDto(ci.id, i.itemNm, i.price, ci.count, im.imgUrl) " +
            "from CartItem ci, ItemImg im " +
            "join ci.item i " +
            "where ci.cart.id = :cartId " +
            "and im.item.id = ci.item.id " +
            "and im.repimgYn = 'Y' " +
            "order by ci.regTime desc"
    )
    List<CartDetailDto> findCartDetailDtoList(Long cartId);
/*
    특정 장바구니 Id에 담긴 상품목록을 조회하면서, 각 상품의 대표이미지를 함께 가져와 등록 시간순으로 정렬
    @Query("SELECT new com.shop.dto.CartDetailDto(ci.id, i.itemNm, i.price, ci.count, im.imgUrl) " +
            "FROM CartItem ci " +
            "JOIN ci.item i " +
            "JOIN ItemImg im ON im.item.id = i.id " +
            "WHERE ci.cart.id = :cartId " +
            "AND im.repimgYn = 'Y' " +
            "ORDER BY ci.regTime DESC")
    List<CartDetailDto> findCartDetailDtoList(@Param("cartId") Long cartId);
    */
}
