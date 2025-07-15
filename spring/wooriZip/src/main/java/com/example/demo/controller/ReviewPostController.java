package com.example.demo.controller;

import com.example.demo.dto.ReviewPostDto;
import com.example.demo.entity.ReviewPost;
import com.example.demo.entity.Users;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.ReviewPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@Controller
@RequiredArgsConstructor
@RequestMapping("/review")
public class ReviewPostController {

    private final ReviewPostService reviewPostService;

    // 리뷰 목록 출력 + 평균평점, 별점분포, 정렬, 페이지네이션
    @GetMapping("/product/{productId}")
    public String getReviewBoard(@PathVariable Long productId,
                                 @RequestParam(defaultValue = "0") int page,
                                 @RequestParam(defaultValue = "createdAt") String sortBy,
                                 Model model,
                                 @AuthenticationPrincipal CustomUserDetails customUserDetails) {

        Users user = customUserDetails != null ? customUserDetails.getUser() : null;
        Page<ReviewPost> reviewPage = reviewPostService.getReviews(productId, page, sortBy);

        Map<Integer, Long> ratingSummary = reviewPostService.getRatingDistribution(productId);
        Double averageRating = reviewPostService.getAverageRating(productId);

        boolean hasWritten = (user != null) && reviewPostService.hasWrittenReview(productId, user.getEmail());

        model.addAttribute("reviewPage", reviewPage);
        model.addAttribute("ratingSummary", ratingSummary);
        model.addAttribute("averageRating", averageRating);
        model.addAttribute("hasWritten", hasWritten);
        model.addAttribute("productId", productId);
        model.addAttribute("sortBy", sortBy);
        model.addAttribute("user", user);

        return "review/review";
    }

    // 리뷰 등록
    @PostMapping("/create")
    public String createReview(@ModelAttribute ReviewPostDto dto,
                               @RequestParam(value = "files", required = false) MultipartFile[] files,
                               @AuthenticationPrincipal CustomUserDetails customUserDetails) throws IOException {
        if (customUserDetails == null) {
            return "redirect:/user/login";
        }
        Users user = customUserDetails.getUser();
        dto.setEmail(user.getEmail());
        dto.setNickname(user.getNickname());
        if (files != null && files.length > 0 && !files[0].isEmpty()) {
            dto.setFiles(List.of(files));
        }
        reviewPostService.saveReview(dto);
        return "redirect:/products/" + dto.getProductId();
    }

    // 리뷰 수정 폼
    @GetMapping("/edit/{id}")
    public String getEditForm(@PathVariable Long id, Model model) {
        ReviewPost review = reviewPostService.getReviewById(id);
        model.addAttribute("review", review);
        model.addAttribute("productId", review.getProduct().getId());
        model.addAttribute("loginUser", review.getEmail()); // 수정 권한 확인을 위해
        return "review/review";
    }

    // 리뷰 수정
    @PostMapping("/update/{id}")
    public String updateReview(@PathVariable Long id,
                               @ModelAttribute ReviewPostDto dto,
                               @RequestParam(value = "files", required = false) MultipartFile[] files,
                               @RequestParam(value = "deleteImages", required = false) List<String> deleteImages,
                               @AuthenticationPrincipal CustomUserDetails customUserDetails) throws IOException {
        if (customUserDetails == null) {
            return "redirect:/user/login";
        }
        Users user = customUserDetails.getUser();
        dto.setEmail(user.getEmail());
        dto.setNickname(user.getNickname());

        if (files != null && files.length > 0 && !files[0].isEmpty()) {
            dto.setFiles(List.of(files));
        }
        reviewPostService.updateReview(id, dto, deleteImages);
        return "redirect:/products/" + dto.getProductId();
    }

    // 리뷰 삭제
    @PostMapping("/delete/{id}")
    public String deleteReview(@PathVariable Long id, @RequestParam Long productId) {
        reviewPostService.deleteReview(id);
        return "redirect:/products/" + productId;
    }
}
