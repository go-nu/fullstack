package com.example.demo.repository;

import com.example.demo.entity.ReviewPost;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ReviewPostRepository extends JpaRepository<ReviewPost, Long> {

    // 페이지네이션 없이 전체 리뷰 조회용
    List<ReviewPost> findByProductId(Long productId);

    // 페이지네이션용
    Page<ReviewPost> findByProductId(Long productId, Pageable pageable);

    boolean existsByProductIdAndEmail(Long productId, String email); // 1인 1리뷰 제한용
}
