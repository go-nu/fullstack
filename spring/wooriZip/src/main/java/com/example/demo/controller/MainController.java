package com.example.demo.controller;

import com.example.demo.dto.InteriorPostDto;
import com.example.demo.entity.Users;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.InteriorPostService;
import com.example.demo.service.RecommendService;
import com.example.demo.service.ReviewPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.client.RestTemplate;

import java.net.http.HttpHeaders;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Controller
@RequiredArgsConstructor
public class MainController {

    private final InteriorPostService interiorPostService;
    private final ReviewPostService reviewPostService;
    private final RecommendService recommendService;

    @GetMapping("/")
    public String welcome(Authentication authentication, Model model) {
        Users user = null;
        if (authentication != null) {
            user = (Users) UserUtils.getUser(authentication);
            model.addAttribute("loginUser", user);
        }

        // ✅ 추천 상품 추가
        if (user != null) {
            try {
                model.addAttribute("products", recommendService.getRecommendedProducts(user));
            } catch (Exception e) {
                model.addAttribute("products", List.of()); // 추천 실패 시 빈 리스트로 대체
            }
        } else {
            model.addAttribute("products", List.of()); // 비로그인 시에도 빈 리스트
        }

        // 전체 게시글 가져오기
        List<InteriorPostDto> allPosts = interiorPostService.findAll();

        // 공지사항과 일반 게시글 분리
        List<InteriorPostDto> notices = allPosts.stream()
                .filter(InteriorPostDto::isNotice)
                .collect(Collectors.toList());

        List<InteriorPostDto> regularPosts = allPosts.stream()
                .filter(post -> !post.isNotice())
                .limit(3)
                .collect(Collectors.toList());

        model.addAttribute("notice", notices.isEmpty() ? null : notices.get(0));
        model.addAttribute("latestInteriorPosts", regularPosts);

        // 최신 리뷰 4개
        model.addAttribute("latestReviews",
                reviewPostService.findAllReviews().stream()
                        .limit(4)
                        .collect(Collectors.toList()));

        return "welcome";
    }

}
