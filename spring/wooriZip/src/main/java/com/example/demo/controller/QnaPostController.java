package com.example.demo.controller;

import com.example.demo.dto.QnaPostDto;
import com.example.demo.entity.Users;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.QnaPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

@Controller
@RequiredArgsConstructor
@RequestMapping("/qna")
public class QnaPostController {

    private final QnaPostService qnaPostService;

    // QnA 등록 (파일 업로드 포함)
    @PostMapping("/create")
    public String create(@RequestParam("productId") Long productId,
                         @ModelAttribute QnaPostDto dto,
                         @RequestParam(value = "files", required = false) MultipartFile[] files,
                         @AuthenticationPrincipal CustomUserDetails customUserDetails) throws IOException {

        if (customUserDetails == null) {
            return "redirect:/user/login";
        }
        Users user = customUserDetails.getUser();
        dto.setProductId(productId);
        dto.setEmail(user.getEmail());
        dto.setNickname(user.getNickname());

        if (files != null && files.length > 0 && !files[0].isEmpty()) {
            dto.setFiles(Arrays.asList(files));
        }

        qnaPostService.saveQna(dto);
        return "redirect:/products/" + productId;
    }

    // QnA 수정 (파일 재업로드 포함)
    @PostMapping("/update/{id}")
    public String update(@PathVariable Long id,
                         @ModelAttribute QnaPostDto dto,
                         @RequestParam(value = "files", required = false) MultipartFile[] files,
                         @AuthenticationPrincipal CustomUserDetails customUserDetails) throws IOException {

        if (customUserDetails == null) {
            return "redirect:/user/login";
        }
        Users user = customUserDetails.getUser();
        dto.setEmail(user.getEmail());
        dto.setNickname(user.getNickname());

        if (files != null && files.length > 0 && !files[0].isEmpty()) {
            dto.setFiles(Arrays.asList(files));
        }

        qnaPostService.updateQna(id, dto);
        return "redirect:/products/" + dto.getProductId();
    }

    // QnA 삭제
    @PostMapping("/delete/{id}")
    public String delete(@PathVariable Long id,
                         @RequestParam Long productId,
                         @AuthenticationPrincipal CustomUserDetails customUserDetails) {
        if (customUserDetails == null) {
            return "redirect:/user/login";
        }
        Users user = customUserDetails.getUser();
        qnaPostService.deleteQna(id, user.getEmail());
        return "redirect:/products/" + productId;
    }

    // QnA 목록 (상품 상세 페이지에서 포함될 부분)
    @GetMapping("/product/{productId}")
    public String qnaList(@PathVariable Long productId,
                          @RequestParam(defaultValue = "0") int page,
                          Model model,
                          @AuthenticationPrincipal CustomUserDetails customUserDetails) {

        Users user = customUserDetails != null ? customUserDetails.getUser() : null;
        List<QnaPostDto> qnaList = qnaPostService.getQnaPostDtoList(productId, page);
        long totalCount = qnaPostService.countByProduct(productId);

        model.addAttribute("qnaList", qnaList);
        model.addAttribute("qnaTotal", totalCount);
        model.addAttribute("qnaPage", page);
        model.addAttribute("isLogin", user != null);
        model.addAttribute("loginUser", user);

        return "product/detail"; // detail.html에서 qna 프래그먼트로 include
    }
}
