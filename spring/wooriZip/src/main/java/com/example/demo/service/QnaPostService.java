package com.example.demo.service;

import com.example.demo.dto.QnaAnswerDto;
import com.example.demo.dto.QnaPostDto;
import com.example.demo.entity.Product;
import com.example.demo.entity.QnaPost;
import com.example.demo.entity.Users;
import com.example.demo.repository.ProductRepository;
import com.example.demo.repository.QnaAnswerRepository;
import com.example.demo.repository.QnaPostRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.transaction.annotation.Transactional;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;
import lombok.Getter;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class QnaPostService {

    private final QnaPostRepository qnaPostRepository;
    private final QnaAnswerRepository qnaAnswerRepository;
    private final ProductRepository productRepository;

    private final String uploadDir = System.getProperty("user.dir") + "/uploads/";

    // Q 등록
    @Transactional
    public void saveQna(QnaPostDto dto) throws IOException {
        Product product = productRepository.findById(dto.getProductId())
                .orElseThrow(() -> new IllegalArgumentException("상품이 존재하지 않습니다."));

        List<String> storedPaths = new ArrayList<>();
        String joinedPaths = "";
        String joinedNames = "";

        if (dto.getFiles() != null && !dto.getFiles().isEmpty()) {
            storedPaths = handleMultipleFiles(dto.getFiles());
            joinedPaths = String.join(",", storedPaths);
            joinedNames = dto.getFiles().stream()
                    .map(MultipartFile::getOriginalFilename)
                    .collect(Collectors.joining(","));
        }

        QnaPost post = QnaPost.builder()
                .title(dto.getTitle())
                .content(dto.getContent())
                .fileNames(joinedNames)
                .filePaths(joinedPaths)
                .email(dto.getEmail())
                .nickname(dto.getNickname())
                .product(product)
                .build();

        qnaPostRepository.save(post);
    }

    // Q 수정
    @Transactional
    public void updateQna(Long id, QnaPostDto dto) throws IOException {
        QnaPost post = qnaPostRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("질문이 존재하지 않습니다."));

        if (!post.getEmail().equals(dto.getEmail())) {
            throw new SecurityException("작성자만 수정할 수 있습니다.");
        }

        deleteFiles(post.getFilePaths());

        List<String> storedPaths = new ArrayList<>();
        String joinedPaths = "";
        String joinedNames = "";

        if (dto.getFiles() != null && !dto.getFiles().isEmpty()) {
            storedPaths = handleMultipleFiles(dto.getFiles());
            joinedPaths = String.join(",", storedPaths);
            joinedNames = dto.getFiles().stream()
                    .map(MultipartFile::getOriginalFilename)
                    .collect(Collectors.joining(","));
        }

        post.setTitle(dto.getTitle());
        post.setContent(dto.getContent());
        post.setFilePaths(joinedPaths);
        post.setFileNames(joinedNames);
        post.setUpdatedAt(LocalDateTime.now());

        qnaPostRepository.save(post);
    }

    // Q 삭제
    @Transactional
    public void deleteQna(Long id, String email) {
        QnaPost post = qnaPostRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("질문이 존재하지 않습니다."));

        if (!post.getEmail().equals(email)) {
            throw new SecurityException("작성자만 삭제할 수 있습니다.");
        }

        deleteFiles(post.getFilePaths());
        qnaPostRepository.delete(post);
    }

    // 파일 저장
    private List<String> handleMultipleFiles(List<MultipartFile> files) throws IOException {
        List<String> filePathList = new ArrayList<>();
        if (files != null) {
            for (MultipartFile file : files) {
                if (!file.isEmpty()) {
                    String fileName = UUID.randomUUID() + "_" + file.getOriginalFilename();
                    File dest = new File(uploadDir + fileName);
                    file.transferTo(dest);
                    filePathList.add("/uploads/" + fileName);
                }
            }
        }
        return filePathList;
    }

    // 파일 삭제
    private void deleteFiles(String filePaths) {
        if (filePaths != null && !filePaths.isEmpty()) {
            String[] paths = filePaths.split(",");
            for (String path : paths) {
                File file = new File(System.getProperty("user.dir") + path);
                if (file.exists()) file.delete();
            }
        }
    }

    // Q 단건 조회
    public QnaPost findById(Long id) {
        return qnaPostRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("질문 없음"));
    }

    // 상품 정보 조회
    public Product getProductById(Long productId) {
        return productRepository.findById(productId)
                .orElseThrow(() -> new IllegalArgumentException("상품을 찾을 수 없습니다."));
    }

    // QnA DTO 리스트 (답변 포함)
    public List<QnaPostDto> getQnaPostDtoList(Long productId, int page, String filter) {
        // 모든 게시글을 가져옵니다
        List<QnaPost> allPosts = qnaPostRepository.findByProductIdOrderByCreatedAtDesc(productId);

        // 필터 적용
        List<QnaPost> filteredPosts = allPosts.stream()
                .filter(post -> {
                    boolean hasAnswer = post.getAnswer() != null;
                    return switch (filter) {
                        case "answered" -> hasAnswer;
                        case "unanswered" -> !hasAnswer;
                        default -> true; // "all" 또는 기타 값
                    };
                })
                .collect(Collectors.toList());

        // 페이지네이션 적용
        int start = page * 5;
        int end = Math.min(start + 5, filteredPosts.size());

        if (start >= filteredPosts.size()) {
            return new ArrayList<>();
        }

        List<QnaPost> pagedPosts = filteredPosts.subList(start, end);

        // DTO 변환
        return pagedPosts.stream()
                .map(post -> {
                    QnaPostDto dto = QnaPostDto.fromEntity(post);
                    qnaAnswerRepository.findByQnaPost(post)
                            .ifPresent(answer -> dto.setAnswer(QnaAnswerDto.fromEntity(answer)));
                    return dto;
                })
                .collect(Collectors.toList());
    }

    // 총 질문 수 (필터 적용)
    public long countByProduct(Long productId, String filter) {
        List<QnaPost> allPosts = qnaPostRepository.findByProductIdOrderByCreatedAtDesc(productId);

        return allPosts.stream()
                .filter(post -> {
                    boolean hasAnswer = post.getAnswer() != null;
                    return switch (filter) {
                        case "answered" -> hasAnswer;
                        case "unanswered" -> !hasAnswer;
                        default -> true; // "all" 또는 기타 값
                    };
                })
                .count();
    }

    // QnA 답변 후 상품상세로 리다이렉트용
    public Long getProductIdByQnaPostId(Long postId) {
        return qnaPostRepository.findById(postId)
                .orElseThrow(() -> new IllegalArgumentException("QnA 게시글이 존재하지 않습니다."))
                .getProduct().getId();
    }

    public Map<String, Long> getQnaStatistics() {
        List<QnaPost> allQna = qnaPostRepository.findAll();

        Map<String, Long> stats = new HashMap<>();
        stats.put("total", (long) allQna.size());
        stats.put("answered", allQna.stream().filter(qna -> qna.getAnswer() != null).count());
        stats.put("unanswered", allQna.stream().filter(qna -> qna.getAnswer() == null).count());

        return stats;
    }

    public Map<String, Map<String, Object>> getQnaByCategoryWithStatus() {
        List<QnaPost> allQna = qnaPostRepository.findAll();
        Map<String, Map<String, Object>> categoryStats = new HashMap<>();

        // 각 상품별로 QnA 게시글을 그룹화하고 생성일 기준 내림차순 정렬
        Map<Long, List<QnaPost>> productQnas = allQna.stream()
                .collect(Collectors.groupingBy(
                        qna -> qna.getProduct().getId(),
                        Collectors.collectingAndThen(
                                Collectors.toList(),
                                list -> {
                                    list.sort((a, b) -> b.getCreatedAt().compareTo(a.getCreatedAt()));
                                    return list;
                                }
                        )
                ));

        allQna.forEach(qna -> {
            String category = qna.getProduct().getCategory().getName();
            categoryStats.putIfAbsent(category, new HashMap<>());

            Map<String, Object> stats = categoryStats.get(category);

            List<Map<String, Object>> answeredList = (List<Map<String, Object>>) stats.getOrDefault("answeredList", new ArrayList<Map<String, Object>>());
            List<Map<String, Object>> unansweredList = (List<Map<String, Object>>) stats.getOrDefault("unansweredList", new ArrayList<Map<String, Object>>());

            // 해당 상품의 QnA 목록에서 현재 게시글의 위치를 찾아 페이지 번호 계산
            List<QnaPost> productQnaList = productQnas.get(qna.getProduct().getId());
            int position = 0;
            for (int i = 0; i < productQnaList.size(); i++) {
                if (productQnaList.get(i).getId().equals(qna.getId())) {
                    position = i;
                    break;
                }
            }

            // 페이지 번호 계산 (5개씩 페이징, 최신글이 1페이지)
            int pageNum = (position / 5) + 1;

            Map<String, Object> postInfo = new HashMap<>();
            postInfo.put("post", qna);
            postInfo.put("page", pageNum);

            if (qna.getAnswer() != null) {
                answeredList.add(postInfo);
            } else {
                unansweredList.add(postInfo);
            }

            // 답변 완료/미답변 목록도 생성일 기준 내림차순 정렬
            Comparator<Map<String, Object>> byCreatedAt = (m1, m2) -> {
                QnaPost p1 = (QnaPost) m1.get("post");
                QnaPost p2 = (QnaPost) m2.get("post");
                return p2.getCreatedAt().compareTo(p1.getCreatedAt());
            };

            answeredList.sort(byCreatedAt);
            unansweredList.sort(byCreatedAt);

            stats.put("answeredList", answeredList);
            stats.put("unansweredList", unansweredList);
            stats.put("answered", answeredList.size());
            stats.put("unanswered", unansweredList.size());
        });

        return categoryStats;
    }

    // QnaPost와 페이지 정보를 함께 담는 내부 클래스
    @Getter
    public class QnaPostWithPage {
        private final QnaPost post;
        private final int page;

        public QnaPostWithPage(QnaPost post, int page) {
            this.post = post;
            this.page = page;
        }
    }

    public List<QnaPost> getUnansweredQna() {
        return qnaPostRepository.findAll().stream()
                .filter(qna -> qna.getAnswer() == null)
                .sorted(Comparator.comparing(QnaPost::getCreatedAt).reversed())
                .collect(Collectors.toList());
    }
}