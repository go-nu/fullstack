package com.example.demo.config;

import com.example.demo.entity.Category;
import com.example.demo.repository.CategoryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CategoryInitializer implements CommandLineRunner {

    private final CategoryRepository categoryRepository;

    @Override
    public void run(String... args) {
        if (categoryRepository.count() == 0) {
            // 1. 대분류 "가구"
            Category furniture = Category.builder()
                    .name("가구")
                    .depth(0)
                    .parent(null)
                    .build();
            categoryRepository.save(furniture);

            // 2. 소분류들 + 세분류 포함 구조
            Map<String, List<String>> subWithChildren = Map.of(
                    "침대", List.of("침대프레임", "침대+매트리스", "침대부속가구"),
                    "테이블.식탁.책상", List.of("식탁", "사무용책상", "좌식책상"),
                    "소파", List.of("일반소파", "좌식소파", "리클라이너"),
                    "서랍.수납장", List.of("서랍", "수납장", "협탁"),
                    "진열장.책장.선반", List.of("진열장", "책장", "선반"),
                    "의자", List.of("학생.사무용의자", "식탁의자", "스툴", "좌식의자"),
                    "행거.붙박이장", List.of("행거", "붙박이장"),
                    "거울", List.of("전신거울", "벽거울", "탁상거울")
            );

            for (Map.Entry<String, List<String>> entry : subWithChildren.entrySet()) {
                String name = entry.getKey();
                List<String> children = entry.getValue();

                // 소분류 생성
                Category mid = Category.builder()
                        .name(name)
                        .depth(1)
                        .parent(furniture)
                        .build();
                categoryRepository.save(mid);

                // 세분류 생성
                for (String childName : children) {
                    Category leaf = Category.builder()
                            .name(childName)
                            .depth(2)
                            .parent(mid)
                            .build();
                    categoryRepository.save(leaf);
                }
            }

            // 3. 기타 소분류 (세분류 없음)
            List<String> etcMids = List.of("선반·진열장·책장");
            for (String name : etcMids) {
                Category mid = Category.builder()
                        .name(name)
                        .depth(1)
                        .parent(furniture)
                        .build();
                categoryRepository.save(mid);
            }
        }
    }

}

