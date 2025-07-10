package com.example.demo.service;

import com.example.demo.dto.QnaAnswerDto;
import com.example.demo.dto.QnaPostDto;
import com.example.demo.entity.Product;
import com.example.demo.entity.QnaPost;
import com.example.demo.repository.ProductRepository;
import com.example.demo.repository.QnaAnswerRepository;
import com.example.demo.repository.QnaPostRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class QnaPostService {

    private final QnaPostRepository qnaPostRepository;
    private final QnaAnswerRepository qnaAnswerRepository;
    private final ProductRepository productRepository;

    private final String uploadDir = System.getProperty("user.dir") + "/uploads/";

    // Q 등록
    public void saveQna(QnaPostDto dto) throws IOException {
        Product product = productRepository.findById(dto.getProductId())
                .orElseThrow(() -> new IllegalArgumentException("상품이 존재하지 않습니다."));

        List<String> storedPaths = handleMultipleFiles(dto.getFiles());
        String joinedPaths = String.join(",", storedPaths);
        String joinedNames = dto.getFiles().stream()
                .map(MultipartFile::getOriginalFilename)
                .collect(Collectors.joining(","));

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
    public void updateQna(Long id, QnaPostDto dto) throws IOException {
        QnaPost post = qnaPostRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("질문이 존재하지 않습니다."));

        if (!post.getEmail().equals(dto.getEmail())) {
            throw new SecurityException("작성자만 수정할 수 있습니다.");
        }

        deleteFiles(post.getFilePaths());

        List<String> storedPaths = handleMultipleFiles(dto.getFiles());
        String joinedPaths = String.join(",", storedPaths);
        String joinedNames = dto.getFiles().stream()
                .map(MultipartFile::getOriginalFilename)
                .collect(Collectors.joining(","));

        post.setTitle(dto.getTitle());
        post.setContent(dto.getContent());
        post.setFilePaths(joinedPaths);
        post.setFileNames(joinedNames);
        post.setUpdatedAt(LocalDateTime.now());

        qnaPostRepository.save(post);
    }

    // Q 삭제
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

    // QnA DTO 리스트 (답변 포함)
    public List<QnaPostDto> getQnaPostDtoList(Long productId, int page) {
        int offset = page * 5;

        List<QnaPost> posts = qnaPostRepository.findByProductIdWithPaging(productId, offset, 5);
        if (posts == null) {
            posts = new ArrayList<>();
        }

        List<QnaPostDto> result = new ArrayList<>();

        for (QnaPost post : posts) {
            QnaPostDto dto = QnaPostDto.fromEntity(post);
            qnaAnswerRepository.findByQnaPost(post).ifPresent(answer ->
                    dto.setAnswer(QnaAnswerDto.fromEntity(answer))
            );
            result.add(dto);
        }

        return result;
    }


    // 총 질문 수
    public long countByProduct(Long productId) {
        return qnaPostRepository.countByProductId(productId);
    }

    // QnA 답변 후 상품상세로 리다이렉트용
    public Long getProductIdByQnaPostId(Long postId) {
        return qnaPostRepository.findById(postId)
                .orElseThrow(() -> new IllegalArgumentException("QnA 게시글이 존재하지 않습니다."))
                .getProduct().getId();
    }

}
