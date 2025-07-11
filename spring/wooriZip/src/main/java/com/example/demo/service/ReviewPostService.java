package com.example.demo.service;

import com.example.demo.dto.ReviewPostDto;
import com.example.demo.entity.Product;
import com.example.demo.entity.ReviewPost;
import com.example.demo.repository.ProductRepository;
import com.example.demo.repository.ReviewPostRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ReviewPostService {

    private final ReviewPostRepository reviewPostRepository;
    private final ProductRepository productRepository;
    private final String uploadDir = System.getProperty("user.dir") + "/uploads/";


    // 리뷰 등록
    public void saveReview(ReviewPostDto dto) throws IOException {
        Product product = productRepository.findById(dto.getProductId())
                .orElseThrow(() -> new IllegalArgumentException("상품이 존재하지 않습니다."));

        List<String> storedPaths = new ArrayList<>();
        String joinedPaths = "";
        String joinedNames = "";

        if (dto.getFiles() != null && !dto.getFiles().isEmpty()) {
            storedPaths = handleMultipleFiles(dto.getFiles());
            joinedPaths = String.join(",", storedPaths);
            joinedNames = dto.getFiles().stream()
                    .map(MultipartFile::getOriginalFilename)
                    .collect(Collectors.joining(","));
        }

        ReviewPost post = ReviewPost.builder()
                .title(dto.getTitle())
                .content(dto.getContent())
                .rating(dto.getRating())
                .fileNames(joinedNames)
                .filePaths(joinedPaths)
                .email(dto.getEmail())
                .nickname(dto.getNickname())
                .product(product)
                .build();

        reviewPostRepository.save(post);
    }

    // 리뷰 수정
    public void updateReview(Long id, ReviewPostDto dto) throws IOException {
        ReviewPost post = reviewPostRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("리뷰가 존재하지 않습니다."));

        deleteFiles(post.getFilePaths());
        List<String> storedPaths = handleMultipleFiles(dto.getFiles());
        String joinedPaths = String.join(",", storedPaths);
        String joinedNames = dto.getFiles().stream()
                .map(MultipartFile::getOriginalFilename)
                .collect(Collectors.joining(","));

        post.setTitle(dto.getTitle());
        post.setContent(dto.getContent());
        post.setRating(dto.getRating());
        post.setFilePaths(joinedPaths);
        post.setFileNames(joinedNames);
        post.setUpdatedAt(LocalDateTime.now());

        reviewPostRepository.save(post);
    }

    // 리뷰 삭제
    public void deleteReview(Long id) {
        ReviewPost post = reviewPostRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("리뷰가 존재하지 않습니다."));
        deleteFiles(post.getFilePaths());
        reviewPostRepository.delete(post);
    }

    // 파일 저장
    private List<String> handleMultipleFiles(List<MultipartFile> files) throws IOException {
        List<String> filePathList = new ArrayList<>();
        if (files != null && !files.isEmpty()) {
            for (MultipartFile file : files) {
                if (file != null && !file.isEmpty()) {
                    String fileName = UUID.randomUUID() + "_" + file.getOriginalFilename();
                    File dest = new File(uploadDir + fileName);
                    file.transferTo(dest);
                    filePathList.add("/uploads/" + fileName);
                }
            }
        }
        return filePathList;
    }

    // 파일 삭제
    private void deleteFiles(String filePaths) {
        if (filePaths != null && !filePaths.isEmpty()) {
            String[] paths = filePaths.split(",");
            for (String path : paths) {
                File file = new File(System.getProperty("user.dir") + path);
                if (file.exists()) file.delete();
            }
        }
    }

    // 평균 평점
    public double getAverageRating(Long productId) {
        List<ReviewPost> list = reviewPostRepository.findByProductId(productId);
        return list.isEmpty() ? 0.0 : list.stream().mapToInt(ReviewPost::getRating).average().orElse(0.0);
    }

    // 별점 분포 (1~5점 카운트)
    public Map<Integer, Long> getRatingDistribution(Long productId) {
        List<ReviewPost> list = reviewPostRepository.findByProductId(productId);
        return list.stream().collect(Collectors.groupingBy(ReviewPost::getRating, Collectors.counting()));
    }

    // 최신순 / 평점순 정렬 + 페이지네이션
    public Page<ReviewPost> getReviews(Long productId, int page, String sortBy) {
        Sort sort = sortBy.equals("rating") ? Sort.by(Sort.Direction.DESC, "rating") : Sort.by(Sort.Direction.DESC, "createdAt");
        Pageable pageable = PageRequest.of(page, 5, sort);
        return reviewPostRepository.findByProductId(productId, pageable);
    }

    // 1인 1리뷰 제한 확인
    public boolean hasWrittenReview(Long productId, String email) {
        return reviewPostRepository.existsByProductIdAndEmail(productId, email);
    }
}
