package com.example.demo.repository;


import com.example.demo.entity.Product;
import com.example.demo.entity.Users;
import com.example.demo.entity.Wishlist;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface WishlistRepository extends JpaRepository<Wishlist, Long> {
    // 사용자 + 상품 조합으로 찜 여부 확인
    boolean existsByUserAndProduct(Users user, Product product);

    // 찜 취소를 위한 조회
    Optional<Wishlist> findByUserAndProduct(Users user, Product product);


}
