package com.example.demo.repository;

import com.example.demo.entity.AttributeValue;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface AttributeValueRepository extends JpaRepository<AttributeValue, Long> {
    // 특정 속성(attribute) id로 값 목록 조회
    List<AttributeValue> findByAttributeId(Long attributeId);
}