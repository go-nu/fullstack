package com.example.firstproject1.service;

import com.example.firstproject1.dto.ArticleForm;
import com.example.firstproject1.entity.Article;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class ArticleServiceTest {
    // 2. ArticleService를 DI
    @Autowired
    ArticleService articleService;

    @Test
    void index() {
        // 예상
        // 모든 게시물을 불러오면 data.sql에 저장했던 데이터들이 불러와 질것이라는 예상.
        Article a = new Article(1L, "가가가가","1111");
        Article b = new Article(2L, "나나나나","2222");
        Article c = new Article(3L, "다다다다","3333");

        // List로 만들기.
        List<Article> expected = new ArrayList<Article>(Arrays.asList(a,b,c));

        // 실제
        List<Article> articles = articleService.index();

        // 비교
        // expected(예상)와 articles(실제) 비교
        assertEquals(expected.toString(), articles.toString());
    }

    @Test
    void show_성공____존재하는_id_입력() {
        // 예상
        Long id = 1L;
        Article expected = new Article(id, "가가가가", "1111");

        // 실제
        Article article = articleService.show(id);

        // 비교
        assertEquals(expected.toString(), article.toString());

    }

    @Test
    void show_실패____존재하지_않는_id_입력() {
        // 예상
        Long id = -1L;
        Article expected = null;
        // ArticleService에서 show를 했을때 데이터가 없으면 null을 리턴하기로 했기때문에 예상값은 null이 되어야한다.
        // ArticleService -> return articleRepository.findById(id).orElse(null);

        // 실제
        Article article = articleService.show(id);

        // 비교
        // null을 toString()을 호출할 수 없음.
        assertEquals(expected, article);
    }

    @Test
    @Transactional
    void create_성공____title과_content만_있는_dto_입력() {
        // 예상
        // title,content를 만들고 이걸 이용해서 dto 생성.
        String title = "라라";
        String content = "44";
        ArticleForm dto = new ArticleForm(null, title, content);
        // 예상 결과 값은 기존에 1,2,3 데이터가 있었기 때문에 id가 4인 객체가 생성될 것이다.
        Article expected = new Article(4L, title, content);

        // 실제
        Article article = articleService.create(dto);

        // 비교
        assertEquals(expected.toString(), article.toString());
    }

    @Test
    void create_실패____id가_포함된_dto_입력() {
        // 예상
        // title,content를 만들고 이걸 이용해서 dto 생성.
        // dto를 만들때 id는 자동생성인데 id를 함께 전달했을때 null이 오도록 코드를 작성해놨다.
        String title = "라라";
        String content = "44";
        ArticleForm dto = new ArticleForm(4L, title, content);

        Article expected = null;
        //if(article.getId() != null) {
        //            return null;
        //        }

        // 실제
        Article article = articleService.create(dto);

        // 비교
        assertEquals(expected, article);
    }

    @Test
    @Transactional
    void update_성공____존재하는_id와_title_content가_있는_dto_입력() {
        // 예상
        // title,content를 만들고 이걸 이용해서 dto 생성.
        Long id = 1L;
        String title = "라라";
        String content = "44";
        ArticleForm dto = new ArticleForm(id, title, content);

        Article expected = new Article(1L, title, content);

        // 실제
        Article article = articleService.update(id,dto);

        // 비교
        assertEquals(expected.toString(), article.toString());
    }

    @Test
    @Transactional
    void update_성공____존재하는_id와_title만_있는_dto_입력() {
        // 예상
        // title,content를 만들고 이걸 이용해서 dto 생성.
        Long id = 1L;
        String title = "라라";
        ArticleForm dto = new ArticleForm(id, title,"ddd");
        Article expected = new Article(1L, title, "ddd");
        // 실제
        Article article = articleService.update(id,dto);

        // 비교
        assertEquals(expected.toString(), article.toString());
    }

    @Test
    @Transactional
    void update_실패____존재하지_않는_id의_dto_입력() {
        // 예상
        Long id = 10L;
        ArticleForm dto = new ArticleForm(id, "ddd","ddd");

        Article expected = null;

        // 실제
        Article article = articleService.update(id,dto);

        // 비교
        assertEquals(expected, article);

    }

    @Test
    @Transactional
    void update_실패____id만_있는_dto_입력() {
        // 예상
        Long id = 1L;
        ArticleForm dto = new ArticleForm(id, "가가가가", "1111");

        Article expected = new Article(id, "가가가가", "1111");

        // 실제
        Article article = articleService.update(id,dto);

        // 비교
        assertEquals(expected.toString(), article.toString());
    }

    @Test
    @Transactional
    void delete_성공____존재하는_id_입력() {
        // 예상
        Long id = 1L;
        Article expected = new Article(id, "가가가가","11111");

        // 실제
        Article article = articleService.delete(id);

        // 비교
        assertEquals(expected.toString(), article.toString());
    }

    @Test
    @Transactional
    void delete_실패____존재하지_않는_id_입력() {
        // 예상
        Long id = 10L;
        Article expected = null;

        // 실제
        Article article = articleService.delete(id);

        // 비교
        assertEquals(expected, article);
    }
}