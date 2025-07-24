package com.example.firstproject1.service;

import com.example.firstproject1.dto.CommentDto;
import com.example.firstproject1.entity.Article;
import com.example.firstproject1.entity.Comment;
import com.example.firstproject1.repository.ArticleRepository;
import com.example.firstproject1.repository.CommentRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Service
@Slf4j
public class CommentService {
    @Autowired
    private CommentRepository commentRepository;

    @Autowired
    private ArticleRepository articleRepository;
    // 댓글이 달린 게시글도 가져와야하기 때문에 ArticleRepository도 함께 가져오기
    public List<CommentDto> comments(Long articleId) {
        // 조회: 댓글 목록 조회(게시글 아이디를 통해 해당 게시글의 댓글 목록 조회)
        List<Comment> comments = commentRepository.findByArticleId(articleId);

        // 변환: 엔티티 -> DTO
        // CommentApiController에서 List<Comment> -> List<CommentDto>로 반환하기로 했기때문에 엔티티 -> DTO로 변환
        List<CommentDto> dtos = new ArrayList<CommentDto>();

        // 비어있는 dtos에다가 댓글들을 변환해서 add하기
        for(int i = 0; i < comments.size(); i++) {
            // comments 값을 하나하나 꺼내서 넣기
            Comment c = comments.get(i);
            CommentDto dto = CommentDto.createCommentDto(c); // Dto로 변환
            dtos.add(dto);
        }

        // 반환
        return dtos;
    }

@Transactional
    public CommentDto create(Long articleId, CommentDto dto) {
    //log.info("입력값 => {}",articleId);
    //log.info("입력값 => {}",dto);
     // 게시글 조회 및 예외 발생
    // .orElseThrow(() -> new IllegalArgumentException()) article이 없다면 예외발생시켜서 다음 코드가 실행되지 않는다.
    Article article = articleRepository.findById(articleId)
            .orElseThrow(() -> new IllegalArgumentException("댓글 생성 실패!! 대상 게시글이 없습니다."));

    // 댓글 엔티티 생성
    Comment comment = Comment.createComment(dto, article);

    // 댓글 엔티티를 DB로 저장
    Comment created = commentRepository.save(comment);

    // DTO로 변환하여 반환
    //return CommentDto.createCommentDto(created);
    CommentDto createDto= CommentDto.createCommentDto(created);
    //log.info("반환값 =>{}",createDto);
    return createDto;
    }
    // 댓글 조회 및 예외 발생
    public CommentDto update(Long id, CommentDto dto) {
       Comment target = commentRepository.findById(id)
                .orElseThrow(()->new IllegalArgumentException("댓글 수정 실패! 대상 댓글이 없습니다."));
    //댓글 수정
        target.patch(dto);

    // 댓글 DB로 갱신
    Comment updated = commentRepository.save(target);

    // 댓글 엔티티를 DTO 로 변환 및  반환
    return  CommentDto.createCommentDto(updated);

    }

    public CommentDto delete(Long id) {
        Comment target = commentRepository.findById(id)
                .orElseThrow(()->new IllegalArgumentException("댓글 삭제 실패! 대상 댓글이 없습니다."));
        //댓글을 db에서 삭제
        commentRepository.delete(target);

        return CommentDto.createCommentDto(target);
    }
}

