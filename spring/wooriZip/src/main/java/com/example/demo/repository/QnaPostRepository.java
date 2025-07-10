package com.example.demo.repository;

import com.example.demo.entity.QnaPost;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface QnaPostRepository extends JpaRepository<QnaPost, Long> {

    //총 문의수
    long countByProductId(Long productId);

    //페이지네이션
    @Query(value = "SELECT * FROM qna_post WHERE product_id = :productId ORDER BY created_at DESC LIMIT :limit OFFSET :offset", nativeQuery = true)
    List<QnaPost> findByProductIdWithPaging(Long productId, int offset, int limit);
}
