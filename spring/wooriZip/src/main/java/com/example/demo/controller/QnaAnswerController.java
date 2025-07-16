package com.example.demo.controller;

import com.example.demo.dto.QnaAnswerDto;
import com.example.demo.entity.QnaAnswer;
import com.example.demo.entity.QnaPost;
import com.example.demo.entity.Users;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.QnaAnswerService;
import com.example.demo.service.QnaPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@Controller
@RequiredArgsConstructor
@RequestMapping("/qna/answer")
public class QnaAnswerController {

    private final QnaAnswerService qnaAnswerService;
    private final QnaPostService qnaPostService;

    // 답변 등록
    @PostMapping("/create/{postId}")
    public String create(@PathVariable Long postId,
                         @RequestParam String content,
                         @RequestParam(required = false) Integer qnaPage,
                         Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        Users user = (Users) UserUtils.getUser(authentication);
        qnaAnswerService.saveAnswer(postId, content, user);
        Long productId = qnaPostService.getProductIdByQnaPostId(postId);

        // 페이지 정보가 있으면 해당 페이지로, 없으면 기본 페이지로
        String pageParam = qnaPage != null ? "?qnaPage=" + qnaPage : "";
        return "redirect:/products/" + productId + pageParam + "#qna-tab";
    }

    // 답변 수정
    @PostMapping("/update/{answerId}")
    public String update(@PathVariable Long answerId,
                         @RequestParam String content,
                         @RequestParam Long productId,
                         @RequestParam(required = false) Integer qnaPage,
                         Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        Users user =  (Users) UserUtils.getUser(authentication);
        qnaAnswerService.updateAnswer(answerId, content, user);

        // 페이지 정보가 있으면 해당 페이지로, 없으면 기본 페이지로
        String pageParam = qnaPage != null ? "?qnaPage=" + qnaPage : "";
        return "redirect:/products/" + productId + pageParam + "#qna-tab";
    }

    // 답변 삭제
    @PostMapping("/delete/{answerId}")
    public String delete(@PathVariable Long answerId,
                         @RequestParam Long productId,
                         @RequestParam(required = false) Integer qnaPage,
                         Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        Users user =  (Users) UserUtils.getUser(authentication);
        qnaAnswerService.deleteAnswer(answerId, user);

        // 페이지 정보가 있으면 해당 페이지로, 없으면 기본 페이지로
        String pageParam = qnaPage != null ? "?qnaPage=" + qnaPage : "";
        return "redirect:/products/" + productId + pageParam + "#qna-tab";
    }

    @GetMapping("/admin/dashboard")
    @PreAuthorize("hasRole('ADMIN')")
    public String adminDashboard(Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        // 전체 통계
        Map<String, Long> statistics = qnaPostService.getQnaStatistics();
        model.addAttribute("statistics", statistics);

        // 카테고리별 QnA 목록 (답변/미답변)
        Map<String, Map<String, Object>> categoryStats = qnaPostService.getQnaByCategoryWithStatus();
        model.addAttribute("categoryStats", categoryStats);

        return "qna/dashboard";
    }

    @GetMapping("/admin/unanswered")
    @PreAuthorize("hasRole('ADMIN')")
    public String getUnansweredQna(Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/login";
        model.addAttribute("unansweredQna", qnaPostService.getUnansweredQna());
        return "qna/admin/unanswered";
    }
}
