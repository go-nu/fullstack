package com.example.demo.repository;

import com.example.demo.entity.ProductModelAttribute;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ProductModelAttributeRepository extends JpaRepository<ProductModelAttribute, Long> {
}