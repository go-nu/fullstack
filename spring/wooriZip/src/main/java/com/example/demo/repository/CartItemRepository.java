package com.example.demo.repository;

import com.example.demo.entity.CartItem;
import com.example.demo.entity.Product;
import com.example.demo.entity.ProductModel;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface CartItemRepository extends JpaRepository<CartItem, Long> {
    Optional<CartItem> findByProductAndProductModel(Product product, com.example.demo.entity.ProductModel productModel);
    void deleteByProduct(Product product); // 0721 해당 상품 관련 product_id가 있는 행 삭제 상품관리
    List<CartItem> findByCartUserEmail(String email);

    @Query("SELECT ci FROM CartItem ci WHERE ci.cart.user.email = :email AND ci.product.isDeleted = false")
    List<CartItem> findValidCartItemsByUserEmail(@Param("email") String email);
}
