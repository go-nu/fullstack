package com.example.demo.controller;

import com.example.demo.service.InteriorPostService;
import com.example.demo.service.ReviewPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
@RequiredArgsConstructor
public class MainController {
    private final InteriorPostService interiorPostService;
    private final ReviewPostService reviewPostService;

    @GetMapping("/")
    public String welcome(Authentication authentication, Model model) {
        if (authentication != null) {
            model.addAttribute("loginUser", UserUtils.getUser(authentication));
        }

        // 최신 인테리어 게시글 4개
        model.addAttribute("latestInteriorPosts", 
            interiorPostService.findAll().stream()
                .limit(4)
                .collect(java.util.stream.Collectors.toList()));

        // 최신 리뷰 4개
        model.addAttribute("latestReviews", 
            reviewPostService.findAllReviews().stream()
                .limit(4)
                .collect(java.util.stream.Collectors.toList()));

        return "welcome";
    }
} 