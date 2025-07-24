package com.example.firstproject1.service;


import com.example.firstproject1.dto.ArticleForm;
import com.example.firstproject1.entity.Article;
import com.example.firstproject1.repository.ArticleRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class ArticleService {

    @Autowired
    private ArticleRepository articleRepository;


    public List<Article> index() {
        return articleRepository.findAll();
    }

    public Article show(Long id) {
        return articleRepository.findById(id).orElse(null);
    }
    @Transactional
    public Article create(ArticleForm dto) {
        Article article = dto.toEntity();
        // 생성할때 id를 넣었으면 null을 리턴
        if(article.getId() != null) {
            return null;
        }
        return  articleRepository.save(article);
    }
    @Transactional
    public Article update(Long id, ArticleForm dto) {

        // dto를 Entity로 변경
        Article article = dto.toEntity();
        // target 찾기
        Article target = articleRepository.findById(id).orElse(null);
        // 잘못된 요청이면 null
        if(target == null || id != article.getId()) {
            return null;
        }
        // 업데이트 후 updated 리턴
        target.patch(article);
        Article updated = articleRepository.save(target);
        return updated;
    }
    @Transactional
    public Article delete(Long id) {
        Article target = articleRepository.findById(id).orElse(null);

        if(target == null) {
            return null;
        }

        articleRepository.delete(target);
        return target;
    }
@Transactional
     // 해당 메소드를 트랜잭션으로 묶는다!
    public List<Article> createArticles(List<ArticleForm> dtos) {
        // dto 묶음을 entity 묶음으로 변환
        List<Article> articleList = dtos.stream().map(dto -> dto.toEntity()).collect(Collectors.toList());

        // 위코드 for문으로 작성
//        List<Article> articleList = new ArrayList<>();
//        for (int i = 0; i < dtos.size(); i++) {
//            ArticleForm dto = dtos.get(i);
//            Article entity = dto.toEntity();
//            articleList.add(entity);
//        }

        // entity 묶음을 DB로 저장
        articleList.stream().forEach(article -> articleRepository.save(article));

        // 위코드 for문으로 작성
//        for (int i = 0; i < articleList.size(); i++) {
//            Article article = articleList.get(i);
//            articleRepository.save(article);
//        }

        // 강제 예외 발생 시키기
        articleRepository.findById(-1L).orElseThrow(
                () -> new IllegalArgumentException("결재 실패")
        );
        // 결과값 반환
        return  articleList;
    }

}
