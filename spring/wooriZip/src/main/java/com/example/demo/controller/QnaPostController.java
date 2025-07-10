package com.example.demo.controller;

import com.example.demo.dto.QnaPostDto;
import com.example.demo.entity.Users;
import com.example.demo.service.QnaPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@Controller
@RequiredArgsConstructor
@RequestMapping("/qna")
public class QnaPostController {

    private final QnaPostService qnaPostService;

    // QnA 게시글 등록
    @PostMapping("/create")
    public String create(@ModelAttribute QnaPostDto dto,
                         @AuthenticationPrincipal Users user) throws IOException {
        dto.setEmail(user.getEmail());
        dto.setNickname(user.getNickname());
        qnaPostService.saveQna(dto);
        return "redirect:/products/" + dto.getProductId();
    }

    // QnA 게시글 수정
    @PostMapping("/update/{id}")
    public String update(@PathVariable Long id,
                         @ModelAttribute QnaPostDto dto,
                         @AuthenticationPrincipal Users user) throws IOException {
        dto.setEmail(user.getEmail());
        dto.setNickname(user.getNickname());
        qnaPostService.updateQna(id, dto);
        return "redirect:/products/" + dto.getProductId();
    }

    // QnA 게시글 삭제
    @PostMapping("/delete/{id}")
    public String delete(@PathVariable Long id,
                         @RequestParam Long productId,
                         @AuthenticationPrincipal Users user) {
        qnaPostService.deleteQna(id, user.getEmail());
        return "redirect:/products/" + productId;
    }

    // 상품 상세 페이지에서 QnA 목록 + 총 개수 모델 전달
    @GetMapping("/product/{productId}")
    public String qnaList(@PathVariable Long productId,
                          @RequestParam(defaultValue = "0") int page,
                          Model model,
                          @AuthenticationPrincipal Users user) {

        List<QnaPostDto> qnaList = qnaPostService.getQnaPostDtoList(productId, page);
        long totalCount = qnaPostService.countByProduct(productId);

        model.addAttribute("qnaList", qnaList);
        model.addAttribute("qnaTotal", totalCount);
        model.addAttribute("qnaPage", page);
        model.addAttribute("isLogin", user != null);

        return "product/detail";  // detail.html에서 qna 프래그먼트로 include
    }
}
