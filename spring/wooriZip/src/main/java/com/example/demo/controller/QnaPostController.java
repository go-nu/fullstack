package com.example.demo.controller;

import com.example.demo.dto.QnaPostDto;
import com.example.demo.entity.Users;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.QnaPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Arrays;

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
                         Authentication authentication) throws IOException {

        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users user = (Users) UserUtils.getUser(authentication);
        dto.setProductId(productId);
        dto.setEmail(user.getEmail());
        dto.setNickname(user.getNickname());

        if (files != null && files.length > 0 && !files[0].isEmpty()) {
            dto.setFiles(Arrays.asList(files));
        }

        qnaPostService.saveQna(dto);
        return "redirect:/products/" + productId + "#qna-tab";
    }

    // QnA 수정 (파일 재업로드 포함)
    @PostMapping("/update/{id}")
    public String update(@PathVariable Long id,
                         @ModelAttribute QnaPostDto dto,
                         @RequestParam(value = "files", required = false) MultipartFile[] files,
                         @RequestParam(required = false) Integer qnaPage,
                         @RequestParam(required = false) String qnaFilter,
                         Authentication authentication) throws IOException {

        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users user = (Users) UserUtils.getUser(authentication);
        dto.setEmail(user.getEmail());
        dto.setNickname(user.getNickname());

        if (files != null && files.length > 0 && !files[0].isEmpty()) {
            dto.setFiles(Arrays.asList(files));
        }

        qnaPostService.updateQna(id, dto);

        // 페이지와 필터 정보 유지
        String pageParam = qnaPage != null ? "?qnaPage=" + qnaPage : "";
        String filterParam = qnaFilter != null ? (pageParam.isEmpty() ? "?" : "&") + "qnaFilter=" + qnaFilter : "";
        return "redirect:/products/" + dto.getProductId() + pageParam + filterParam + "#qna-tab";
    }

    // QnA 삭제
    @PostMapping("/delete/{id}")
    public String delete(@PathVariable Long id,
                         @RequestParam Long productId,
                         @RequestParam(required = false) Integer qnaPage,
                         @RequestParam(required = false) String qnaFilter,
                         Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users user = (Users) UserUtils.getUser(authentication);
        qnaPostService.deleteQna(id, user.getEmail());

        // 페이지와 필터 정보 유지
        String pageParam = qnaPage != null ? "?qnaPage=" + qnaPage : "";
        String filterParam = qnaFilter != null ? (pageParam.isEmpty() ? "?" : "&") + "qnaFilter=" + qnaFilter : "";
        return "redirect:/products/" + productId + pageParam + filterParam + "#qna-tab";
    }
}
