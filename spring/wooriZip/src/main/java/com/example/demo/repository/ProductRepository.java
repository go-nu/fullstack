package com.example.demo.repository;


import com.example.demo.entity.Category;
import com.example.demo.entity.Product;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface ProductRepository extends JpaRepository<Product, Long> {
    // 카테고리로 상품 목록 조회
    List<Product> findByCategory(Category category);

    // 선택적으로 소분류 ID로 조회
    List<Product> findByCategory_Id(Long categoryId);

    List<Product> findByCategoryIdIn(List<Long> categoryIds);

    @Query("SELECT p FROM Product p " +
            "LEFT JOIN FETCH p.category c " +
            "LEFT JOIN FETCH c.parent cp " +
            "LEFT JOIN FETCH cp.parent cpp " +
            "WHERE p.id = :id")
    Product findWithCategoryTreeById(@Param("id") Long id);

    List<Product> findByIdIn(List<Long> myWishList);
    
    // 챗봇용 메서드들
    List<Product> findByNameContainingIgnoreCase(String name);
    
    @Query("SELECT p FROM Product p ORDER BY p.price ASC")
    List<Product> findTop5ByOrderByPriceAsc(Pageable pageable);
    
    @Query("SELECT p FROM Product p ORDER BY p.createdAt DESC")
    List<Product> findTop5ByOrderByCreatedAtDesc(Pageable pageable);
    
    // 기본 메서드들 (Pageable 없이)
    default List<Product> findTop5ByOrderByPriceAsc() {
        return findTop5ByOrderByPriceAsc(Pageable.ofSize(5));
    }
    
    default List<Product> findTop5ByOrderByCreatedAtDesc() {
        return findTop5ByOrderByCreatedAtDesc(Pageable.ofSize(5));
    }
}
