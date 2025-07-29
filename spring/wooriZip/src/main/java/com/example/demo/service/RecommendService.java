package com.example.demo.service;

import com.example.demo.entity.Product;
import com.example.demo.entity.ProductModel;
import com.example.demo.entity.ProductModelAttribute;
import com.example.demo.entity.Users;
import com.example.demo.repository.ProductModelAttributeRepository;
import com.example.demo.repository.ProductRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class RecommendService {

    private final RestTemplate restTemplate;
    private final ProductRepository productRepository;
    private final ProductModelAttributeRepository productModelAttributeRepository;

    @Value("${recommend.api.url:http://localhost:8000/recommend}")
    private String recommendApiUrl;

    public List<Product> getRecommendedProducts(Users user) {
        int gender = user.getGender().equalsIgnoreCase("male") ? 1 : 0;
        int ageGroup = calculateAgeGroup(user.getBirth());
        int residenceType = user.getResidenceType();

        int color = getMostUsedOption("색상");
        int size = getMostUsedOption("사이즈");
        int material = getMostUsedOption("소재");

        Map<String, Object> requestBody = Map.of(
                "user_id", user.getId(),
                "gender", gender,
                "age_group", ageGroup,
                "residence_type", residenceType,
                "색상", color,
                "사이즈", size,
                "소재", material
        );

        // ✅ 전달한 입력 로그 출력
        System.out.println("📤 [FastAPI 요청값]");
        requestBody.forEach((key, value) ->
                System.out.println(" - " + key + ": " + value)
        );

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

        ResponseEntity<RecommendResult[]> response = restTemplate.postForEntity(
                recommendApiUrl, request, RecommendResult[].class);

        RecommendResult[] results = response.getBody();

        // ✅ 응답 결과 로그 출력
        System.out.println("📥 [FastAPI 응답값]");
        if (results != null && results.length > 0) {
            for (RecommendResult r : results) {
                System.out.println(" → product_id: " + r.getProduct_id() + ", score: " + r.getScore());
            }
        } else {
            System.out.println(" → 추천 결과 없음 (빈 배열)");
        }

        List<Long> productIds = Arrays.stream(results)
                .map(RecommendResult::getProduct_id)
                .toList();

        return productRepository.findAllById(productIds);
    }

    private int getMostUsedOption(String attributeName) {
        List<ProductModelAttribute> all = productModelAttributeRepository.findAll();

        return all.stream()
                .filter(attr -> attributeName.equals(attr.getAttributeValue().getAttribute().getName()))
                .collect(Collectors.groupingBy(
                        attr -> attr.getAttributeValue().getId(),
                        Collectors.counting()
                ))
                .entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(0L)
                .intValue();
    }

    private int calculateAgeGroup(LocalDate birth) {
        if (birth == null) return 0;
        int age = LocalDate.now().getYear() - birth.getYear();
        if (age < 20) return 0;
        if (age < 30) return 1;
        if (age < 40) return 2;
        return 3;
    }

    public List<Product> getBestProducts() {
        return productRepository.findTopRecommendedProducts();
    }

    public static class RecommendResult {
        private Long product_id;
        private double score;
        public Long getProduct_id() { return product_id; }
        public void setProduct_id(Long id) { this.product_id = id; }
        public double getScore() { return score; }
        public void setScore(double score) { this.score = score; }
    }
}