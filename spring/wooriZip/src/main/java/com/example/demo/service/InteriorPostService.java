package com.example.demo.service;

import com.example.demo.dto.InteriorPostDto;
import com.example.demo.entity.InteriorPost;
import com.example.demo.entity.Users;
import com.example.demo.repository.InteriorPostRepository;
import com.example.demo.repository.LoginRepository;
import com.example.demo.repository.PostCommentRepository;
import com.example.demo.repository.PostLikeRepository;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class InteriorPostService {

    private final InteriorPostRepository repository;
    private final LoginRepository loginRepository;
    private final PostCommentRepository postCommentRepository;
    private final PostLikeRepository postLikeRepository;
    private final PostLikeService postLikeService;

    /** 게시글 전체 목록 조회 */
    public List<InteriorPostDto> findAll() {
        return repository.findAllByOrderByCreatedAtDesc()
                .stream()
                .map(this::toDto)
                .collect(Collectors.toList());
    }

    /** 게시글 단건 조회 */
    public InteriorPostDto findById(Long id, Users loginUser) {
        InteriorPost post = repository.findById(id)
                .orElseThrow(() -> new RuntimeException("게시글 없음"));

        InteriorPostDto dto = toDto(post);

        // 좋아요 눌렀는지 여부 체크
        boolean liked = postLikeService.hasUserLikedPost(post, loginUser);
        dto.setLikedByCurrentUser(liked);

        return dto;
    }


    /** 게시글 저장/수정 */
    public void save(InteriorPostDto dto) {
        InteriorPost post;

        if (dto.getPostId() != null) {
            post = repository.findById(dto.getPostId()).orElse(null);
            if (post == null) return;

            post.setTitle(dto.getTitle());
            post.setContent(dto.getContent());

            // 새 이미지가 있다면 업데이트, 없으면 기존 이미지 유지
            if (dto.getFileNames() != null && !dto.getFileNames().isEmpty()
                    && dto.getFilePaths() != null && !dto.getFilePaths().isEmpty()) {
                post.setFileName(dto.getFileNames());
                post.setFilePath(dto.getFilePaths());
            } else {
                // 이미지 다 삭제했을 경우 경로 null 처리
                post.setFileName(null);
                post.setFilePath(null);
            }
            // else: 기존 이미지 유지
        } else {
            // 새 글 등록
            post = toEntity(dto);
        }

        repository.save(post);
    }

    /** 게시글 삭제 */
    @Transactional
    public void delete(Long id) {
        InteriorPost post = repository.findById(id)
                .orElseThrow(() -> new RuntimeException("게시글 없음"));

        postCommentRepository.deleteByPost(post); // 댓글 먼저 삭제
        postLikeRepository.deleteByPost(post);    // 좋아요 먼저 삭제
        repository.delete(post);                  // 그다음 게시글 삭제
    }


    /** 조회수 증가 */
    public void increaseViews(Long id) {
        InteriorPost post = repository.findById(id).orElse(null);
        if (post != null) {
            post.setViews(post.getViews() + 1);
            repository.save(post);
        }
    }


    /** 페이지네이션 */
    public Page<InteriorPostDto> findPagedPosts(int page, int size) {
        Pageable pageable = PageRequest.of(page - 1, size, Sort.by(Sort.Direction.DESC, "postId"));
        return repository.findAll(pageable).map(InteriorPostDto::fromEntity);
    }


    // -----------------------
    // Entity <-> DTO 변환
    // -----------------------
    private InteriorPostDto toDto(InteriorPost post) {
        List<String> filePathList = null;
        if (post.getFilePath() != null && !post.getFilePath().isEmpty()) {
            filePathList = Arrays.stream(post.getFilePath().split(","))
                    .map(String::trim)
                    .collect(Collectors.toList());
        }

        return InteriorPostDto.builder()
                .postId(post.getPostId())
                .title(post.getTitle())
                .content(post.getContent())
                .fileNames(post.getFileName())
                .filePaths(post.getFilePath())
                .filePathList(filePathList)  // 뷰에서 반복문으로 출력
                .createdAt(post.getCreatedAt())
                .updatedAt(post.getUpdatedAt())
                .email(post.getUser().getEmail())
                .nickname(post.getUser().getNickname())
                .liked(post.getLiked())
                .views(post.getViews())
                .build();
    }

    private InteriorPost toEntity(InteriorPostDto dto) {
        Users user = loginRepository.findByEmail(dto.getEmail())
                .orElseThrow(() -> new RuntimeException("작성자(User) 정보 없음"));

        InteriorPost post = new InteriorPost();
        post.setTitle(dto.getTitle());
        post.setContent(dto.getContent());
        post.setFileName(dto.getFileNames());
        post.setFilePath(dto.getFilePaths());
        post.setLiked(0);
        post.setViews(0);
        post.setUser(user); // 작성자 설정

        return post;
    }

}