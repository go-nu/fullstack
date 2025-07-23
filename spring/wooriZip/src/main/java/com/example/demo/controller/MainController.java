package com.example.demo.controller;

import com.example.demo.dto.InteriorPostDto;
import com.example.demo.service.InteriorPostService;
import com.example.demo.service.ReviewPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

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

        // 전체 게시글 가져오기
        List<InteriorPostDto> allPosts = interiorPostService.findAll();

        // 공지사항과 일반 게시글 분리
        List<InteriorPostDto> notices = allPosts.stream()
            .filter(post -> post.isNotice())
            .collect(Collectors.toList());

        List<InteriorPostDto> regularPosts = allPosts.stream()
            .filter(post -> !post.isNotice())
            .limit(3)  // 일반 게시글은 3개만
            .collect(Collectors.toList());

        // 공지사항이 있으면 첫 번째 것만, 없으면 null
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