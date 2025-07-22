package com.example.demo.controller;

import com.example.demo.dto.InteriorPostDto;
import com.example.demo.dto.PostCommentDto;
import com.example.demo.dto.ReviewPostDto;
import com.example.demo.entity.Users;
import com.example.demo.constant.Role;
import com.example.demo.service.InteriorPostService;
import com.example.demo.service.PostCommentService;
import com.example.demo.service.PostLikeService;
import com.example.demo.service.ReviewPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Controller
@RequiredArgsConstructor
@RequestMapping("/interior")
public class InteriorPostController {

    private final InteriorPostService service;
    private final PostCommentService commentService;
    private final PostLikeService postLikeService;
    private final ReviewPostService reviewPostService;

    /** 글 목록 + 페이지네이션 추가 */
    @GetMapping
    public String list(@RequestParam(defaultValue = "1") int page,
                       Model model,
                       Authentication authentication) {
        // 비로그인 사용자도 목록 조회 가능
        String email = UserUtils.getEmail(authentication);
        Users loginUser = email != null ? (Users) UserUtils.getUser(authentication) : null;
        model.addAttribute("loginUser", loginUser);

        int pageSize = 10;
        Page<InteriorPostDto> postPage = service.findPagedPosts(page, pageSize);

        model.addAttribute("postPage", postPage);
        model.addAttribute("currentPage", page);
        model.addAttribute("totalPages", postPage.getTotalPages());

        return "interior/list";
    }

    /** 글 작성 폼 */
    @GetMapping("/write")
    public String writeForm(Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        model.addAttribute("loginUser", UserUtils.getUser(authentication));

        model.addAttribute("dto", new InteriorPostDto());
        return "interior/write";
    }

    /** 글 작성 처리 + 파일 업로드 포함 */
    @PostMapping("/write")
    @ResponseBody
    public String writePost(@ModelAttribute InteriorPostDto dto,
                            @RequestParam(value = "files", required = false) MultipartFile[] files,
                            @RequestParam(value = "isNotice", required = false) Boolean isNotice,
                            Authentication authentication) {

        String email = UserUtils.getEmail(authentication);
        if (email == null) return "unauthorized";

        Users loginUser = (Users) UserUtils.getUser(authentication);
        dto.setEmail(loginUser.getEmail());
        dto.setNickname(loginUser.getNickname());

        // 공지사항 권한 체크
        if (Boolean.TRUE.equals(isNotice)) {
            if (loginUser.getRole() != Role.ADMIN) {
                return "forbidden";
            }
            dto.setNotice(true);
        }

        if (files != null && files.length > 0 && !files[0].isEmpty()) {
            handleMultipleFiles(dto, files);
        }

        service.save(dto);
        return "success";
    }

    /** 상세 보기 + 조회수 증가 + 댓글 조회 */
    @GetMapping("/{id}")
    public String detail(@PathVariable Long id,
                         Model model,
                         Authentication authentication) {
        // 비로그인 사용자도 상세 조회 가능
        String email = UserUtils.getEmail(authentication);
        Users loginUser = email != null ? (Users) UserUtils.getUser(authentication) : null;
        model.addAttribute("loginUser", loginUser);

        service.increaseViews(id);
        InteriorPostDto dto = service.findById(id, loginUser);

        model.addAttribute("dto", dto);
        model.addAttribute("commentList", commentService.findByPostId(id));
        model.addAttribute("loginUser", loginUser);

        return "interior/detail";
    }

    /** 수정 폼 */
    @GetMapping("/edit/{id}")
    public String editForm(@PathVariable Long id,
                           Model model,
                           Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);
        model.addAttribute("loginUser", loginUser);
        model.addAttribute("dto", service.findById(id, loginUser));

        return "interior/edit";
    }

    /** 수정 */
    @PostMapping("/edit")
    @ResponseBody
    public String edit(@ModelAttribute InteriorPostDto dto,
                       @RequestParam(value = "files", required = false) MultipartFile[] files,
                       @RequestParam(value = "deleteIndexes", required = false) List<Integer> deleteIndexes,
                       Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);

        dto.setEmail(loginUser.getEmail());
        dto.setNickname(loginUser.getNickname());

        // 기존 이미지 목록
        List<String> existingPaths = new ArrayList<>();
        List<String> existingNames = new ArrayList<>();
        if (dto.getFilePaths() != null && !dto.getFilePaths().isEmpty()) {
            existingPaths = new ArrayList<>(List.of(dto.getFilePaths().split(",")));
        }
        if (dto.getFileNames() != null && !dto.getFileNames().isEmpty()) {
            existingNames = new ArrayList<>(List.of(dto.getFileNames().split(",")));
        }

        // 삭제 대상 인덱스 제거
        if (deleteIndexes != null && !deleteIndexes.isEmpty()) {
            deleteIndexes.sort(Comparator.reverseOrder());
            for (int index : deleteIndexes) {
                if (index >= 0 && index < existingPaths.size()) {
                    // 파일 실제 삭제
                    String path = existingPaths.get(index);
                    File file = new File(System.getProperty("user.dir") + path);
                    if (file.exists()) file.delete();
                    // 리스트에서 제거
                    existingPaths.remove(index);
                    if (index < existingNames.size()) {
                        existingNames.remove(index);
                    }
                }
            }
        }

        // 새 이미지 업로드
        List<String> addedPaths = new ArrayList<>();
        List<String> addedNames = new ArrayList<>();
        if (files != null && files.length > 0 && !files[0].isEmpty()) {
            List<String>[] result = handleAndReturnFiles(files);
            addedPaths = result[0];
            addedNames = result[1];
        }

        // 병합 저장
        existingPaths.addAll(addedPaths);
        existingNames.addAll(addedNames);
        dto.setFilePaths(String.join(",", existingPaths));
        dto.setFileNames(String.join(",", existingNames));

        service.save(dto);
        return "success";
    }

    /** 삭제 */
    @PostMapping("/delete/{id}")
    public String delete(@PathVariable Long id,
                         @RequestParam(required = false) Boolean fromMyPage,
                         Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";

        service.delete(id);

        if (Boolean.TRUE.equals(fromMyPage)) {
            return "redirect:/user/mypage";
        }
        return "redirect:/interior";
    }

    /** 댓글 등록 */
    @PostMapping("/{postId}/comment")
    public String addComment(@PathVariable Long postId,
                             @RequestParam String content,
                             Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);

        PostCommentDto comment = PostCommentDto.builder()
                .postId(postId)
                .email(loginUser.getEmail())
                .nickname(loginUser.getNickname())
                .content(content)
                .build();

        commentService.save(comment);
        return "redirect:/interior/" + postId;
    }

    /** 댓글 삭제 */
    @PostMapping("/{postId}/comment/delete/{commentId}")
    public String deleteComment(@PathVariable Long postId,
                                @PathVariable Long commentId,
                                Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);

        PostCommentDto dto = commentService.findById(commentId);
        if (!dto.getEmail().equals(loginUser.getEmail())) {
            return "redirect:/interior/" + postId;
        }

        commentService.delete(commentId);
        return "redirect:/interior/" + postId;
    }

    /** 댓글 수정 */
    @PostMapping("/{postId}/comment/edit/{commentId}")
    public String editComment(@PathVariable Long postId,
                              @PathVariable Long commentId,
                              @RequestParam String content,
                              Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);

        PostCommentDto dto = commentService.findById(commentId);
        if (!dto.getEmail().equals(loginUser.getEmail())) {
            return "redirect:/interior/" + postId;
        }

        dto.setContent(content);
        commentService.update(dto);
        return "redirect:/interior/" + postId;
    }

    /** 파일 업로드 처리 */
    private void handleMultipleFiles(InteriorPostDto dto, MultipartFile[] files) {
        if (files == null || files.length == 0) return;

        List<String>[] result = handleAndReturnFiles(files);
        dto.setFilePaths(String.join(",", result[0]));
        dto.setFileNames(String.join(",", result[1]));
    }

    /** 파일 업로드 처리 및 경로 반환 */
    private List<String>[] handleAndReturnFiles(MultipartFile[] files) {
        List<String> filePaths = new ArrayList<>();
        List<String> fileNames = new ArrayList<>();
        String uploadDir = System.getProperty("user.dir") + "/uploads/";

        File dir = new File(uploadDir);
        if (!dir.exists()) dir.mkdirs();

        for (MultipartFile file : files) {
            if (!file.isEmpty()) {
                try {
                    String originalName = file.getOriginalFilename();
                    String uniqueName = System.currentTimeMillis() + "_" + originalName;
                    File dest = new File(uploadDir + uniqueName);
                    file.transferTo(dest);

                    filePaths.add("/uploads/" + uniqueName);
                    fileNames.add(uniqueName);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return new List[]{filePaths, fileNames};
    }

    /** 좋아요 */
    @PostMapping("/{postId}/like")
    @ResponseBody
    public String toggleLike(@PathVariable Long postId,
                             Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);

        return postLikeService.toggleLike(postId, loginUser.getEmail());
    }

}