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
import org.springframework.security.core.Authentication;
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
                                 Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        Users user = email != null ? (Users) UserUtils.getUser(authentication) : null;
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
                               Authentication authentication) throws IOException {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        Users user = (Users) UserUtils.getUser(authentication);
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
    public String getEditForm(@PathVariable Long id, Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        
        ReviewPost review = reviewPostService.getReviewById(id);
        // 작성자 본인 확인
        if (!review.getEmail().equals(email)) {
            return "redirect:/access-denied";
        }
        
        model.addAttribute("review", review);
        model.addAttribute("productId", review.getProduct().getId());
        model.addAttribute("loginUser", UserUtils.getUser(authentication));
        return "review/review";
    }

    // 리뷰 수정
    @PostMapping("/update/{id}")
    public String updateReview(@PathVariable Long id,
                               @ModelAttribute ReviewPostDto dto,
                               @RequestParam(value = "files", required = false) MultipartFile[] files,
                               @RequestParam(value = "deleteImages", required = false) List<String> deleteImages,
                               Authentication authentication) throws IOException {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        Users user = (Users) UserUtils.getUser(authentication);
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
    public String deleteReview(@PathVariable Long id, 
                             @RequestParam Long productId,
                             Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        
        ReviewPost review = reviewPostService.getReviewById(id);
        // 작성자 본인 확인
        if (!review.getEmail().equals(email)) {
            return "redirect:/access-denied";
        }
        
        reviewPostService.deleteReview(id);
        return "redirect:/products/" + productId;
    }
}
