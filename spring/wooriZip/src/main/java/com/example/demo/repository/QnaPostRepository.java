package com.example.demo.repository;

import com.example.demo.entity.QnaPost;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface QnaPostRepository extends JpaRepository<QnaPost, Long> {

    //총 문의수
    @Query("SELECT COUNT(q) FROM QnaPost q WHERE q.product.id = :productId")
    long countByProductId(@Param("productId") Long productId);

    @Query("SELECT COUNT(q) FROM QnaPost q WHERE q.product.id = :productId AND q.createdAt >= (SELECT p.createdAt FROM QnaPost p WHERE p.id = :postId)")
    long countPositionInProduct(@Param("postId") Long postId, @Param("productId") Long productId);

    //페이지네이션
    @Query(value = "SELECT * FROM qna_post WHERE product_id = :productId ORDER BY created_at DESC LIMIT :limit OFFSET :offset", nativeQuery = true)
    List<QnaPost> findByProductIdWithPaging(Long productId, int offset, int limit);
}
