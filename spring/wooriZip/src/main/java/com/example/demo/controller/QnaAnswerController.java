package com.example.demo.controller;

import com.example.demo.entity.Users;
import com.example.demo.service.QnaAnswerService;
import com.example.demo.service.QnaPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

@Controller
@RequiredArgsConstructor
@RequestMapping("/qna/answer")
public class QnaAnswerController {

    private final QnaAnswerService answerService;
    private final QnaPostService postService;

    // 답변 등록
    @PostMapping("/create/{postId}")
    public String create(@PathVariable Long postId,
                         @RequestParam String content,
                         @AuthenticationPrincipal Users user) {
        answerService.saveAnswer(postId, content, user);
        Long productId = postService.getProductIdByQnaPostId(postId);
        return "redirect:/products/" + productId;
    }

    // 답변 수정
    @PostMapping("/update/{answerId}")
    public String update(@PathVariable Long answerId,
                         @RequestParam String content,
                         @RequestParam Long productId,
                         @AuthenticationPrincipal Users user) {
        answerService.updateAnswer(answerId, content, user);
        return "redirect:/products/" + productId;
    }

    // 답변 삭제
    @PostMapping("/delete/{answerId}")
    public String delete(@PathVariable Long answerId,
                         @RequestParam Long productId,
                         @AuthenticationPrincipal Users user) {
        answerService.deleteAnswer(answerId, user);
        return "redirect:/products/" + productId;
    }
}
