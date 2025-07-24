package com.example.firstproject1.repository;

import com.example.firstproject1.entity.Article;
import com.example.firstproject1.entity.Comment;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DataJpaTest //jpa와 연동 테스트
class CommentRepositoryTest {
    @Autowired
    CommentRepository commentRepository;

    @Test
    @DisplayName("특정 게시글의 모든 댓글 조회")
    void findByArticleId() {
        /*Case 1 : 4번게시글의 모든 댓글 조회 */
        //입력 데이터 준비
        Long articleId = 4L;
        //실제 수행
        List<Comment> comments = commentRepository.findByArticleId(articleId);
        //예상
        Article article = new Article(4L,"당신의 인생 영화는?","댓글 고");

        Comment a = new Comment(1L, article, "Park", "굿 윌 헌팅");
        Comment b = new Comment(2L, article, "Kim", "아이 엠 샘");
        Comment c = new Comment(3L, article, "Choi", "쇼생크 탈출");
        List<Comment> expected = Arrays.asList(a,b,c);
        //비교및 검증
        System.out.println("expeted : "+ expected.toString());
        System.out.println("comments : "+ comments.toString());
        assertEquals(expected.toString(), comments.toString(),"4번글의 모든 댓글을 출력");

    }

    @Test
    @DisplayName("특정 닉네임의 모든 댓글 조회")
    void findByNickname() {
        /*Case 1 : "Park"의 모든 댓글 조회*/
        {
            //입력데이터 준비
            String nickname = "Park";
            // 2. 실제 데이터
            List<Comment> comments = commentRepository.findByNickname(nickname);
            //3. 예상 데이터
            Comment a = new Comment(1L, new Article(4L, "당신의 인생 영화는?", "댓글 고"), nickname, "굿 윌 헌팅");
            Comment b = new Comment(4L, new Article(5L, "당신의 소울 푸드는?", "댓글 고고"),
                    nickname, "치킨");
            Comment c = new Comment(7L, new Article(6L, "당신의 취미는?", "댓글 고고고"),
                    nickname, "조깅");
            List<Comment> expected = Arrays.asList(a, b, c);
            //4.비교검증
            assertEquals(expected.toString(), comments.toString(), "Park의 모든 댓글을 출력!");
            System.out.println(comments.toString());
        }
        /* Case 2: "Kim"의 모든 댓글 조회 */
        {

        }

    }
}