package com.shop.repository;

import com.shop.dto.CartDetailDto;
import com.shop.entity.CartItem;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface CartItemRepository extends JpaRepository<CartItem, Long> {

    CartItem findByCartIdAndItemId(Long cartId, Long itemId);

    @Query("select new com.shop.dto.CartDetailDto(ci.id, i.itemNm, i.price, ci.count, im.imgUrl) " + "from CartItem ci, ItemImg im " + "join ci.item i " + "where ci.cart.id = :cartId " + "and im.item.id = ci.item.id " + "and im.repImgYn = 'Y' " + "order by ci.regTime desc")
    List<CartDetailDto> findCartDetailDtoList(Long cartId);
    /*
    특정 장바구니 Id에 담긴 상품목록을 조회하면서, 각 상품의 대표이미지를 함께 가져와 등록 시간순으로 정렬
    cartItem -> item <- itemImg
    cartItem - itemImg 직접 연결 x, item을 통한 간접 연결
    JOIN ci.item i로 상품을 연결,
    이미지(itemImg)는 WHERE 에서 조건으로 연결
    @Query("SELECT new com.shop.dto.CartDetailDto(ci.id, i.itemNm, i.price, ci.count, im.imgUrl) " +
            "FROM CartItem ci " +
            "JOIN ci.item i " +
            "JOIN ItemImg im ON im.item.id = i.id " +
            "WHERE ci.cart.id = :cartId " +
            "AND im.repimgYn = 'Y' " +
            "ORDER BY ci.regTime DESC")
    List<CartDetailDto> findCartDetailDtoList(@Param("cartId") Long cartId);
    */

    /*
    Entity -> service에서 mapping을 수행하려면,
    @Query("select ci from CartItem ci " +
            "join fetch ci.item i " +
            "where ci.cart.id = :cartId " +
            "order by ci.regTime desc")
    List<CartItem> findCartItemsWithItem(@Param("cartId") Long cartId);

    @Transactional(readOnly = true)
    public List<CartDetailDto> getCartDetailList(String email) {
        Member member = memberRepository.findByEmail(email).orElseThrow(() -> new EntityNotFoundException("회원이 존재하지 않습니다."));

        Cart cart = cartRepository.findByMemberId(member.getId());

        if (cart == null) {
            return Collections.emptyList();
        }

        List<CartItem> cartItems = cartItemRepository.findCartItemsWithItem(cart.getId());

        return cartItems.stream().map(ci -> {
            // 대표 이미지 가져오기
            ItemImg repImg = itemImgRepository.findFirstByItemIdAndRepimgYn(ci.getItem().getId(), "Y");
            return new CartDetailDto(ci.getId(), ci.getItem().getItemNm(), ci.getItem().getPrice(), ci.getCount(), repImg != null ? repImg.getImgUrl() : null);
        }).collect(Collectors.toList());
    }
    */
}
