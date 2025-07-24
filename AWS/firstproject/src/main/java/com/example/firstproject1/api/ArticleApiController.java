package com.example.firstproject1.api;

import com.example.firstproject1.dto.ArticleForm;
import com.example.firstproject1.entity.Article;
import com.example.firstproject1.repository.ArticleRepository;
import com.example.firstproject1.service.ArticleService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@Slf4j
public class ArticleApiController {

    @Autowired
    private ArticleService articleService;

    // GET
    // 전체게시글 불러오기
    @GetMapping("/api/articles")
    public List<Article> index() {
        return articleService.index();
    }

    // 게시글 하나 불러오기
    @GetMapping("/api/articles/{id}")
    public Article show(@PathVariable Long id) {
        return articleService.show(id);
    }

    // POST
    @PostMapping("/api/articles")
    public ResponseEntity<Article> create(@RequestBody ArticleForm dto) {
        Article created = articleService.create(dto);
        // 잘 생성된경우 GOOD : 안됐을때 BAD
        return (created != null) ? ResponseEntity.status(HttpStatus.OK).body(created) : ResponseEntity.status(HttpStatus.BAD_REQUEST).build() ;
    }
    // PATCH
    @PatchMapping("/api/articles/{id}")
    public ResponseEntity<Article> update(@PathVariable Long id,
                                          @RequestBody ArticleForm dto) {

    Article updated = articleService.update(id, dto);
    return (updated != null) ?
            ResponseEntity.status(HttpStatus.OK).body(updated) :
            ResponseEntity.status(HttpStatus.BAD_REQUEST).build();

    }
    @DeleteMapping("/api/articles/{id}")
    public ResponseEntity<Article> delete(@PathVariable Long id) {

        Article deleted = articleService.delete(id);
        return (deleted != null) ?
                ResponseEntity.status(HttpStatus.NO_CONTENT).build() :
                ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
    }
    // 트랜잭션(반드시 성공해야할 일련의 과정) -> 성공못하면 롤백!!
    @PostMapping("/api/transaction-test")
    public ResponseEntity<List<Article>> transactionTest(@RequestBody List<ArticleForm> dtos) {
        List<Article> createsList = articleService.createArticles(dtos);
        return (createsList != null) ?
                ResponseEntity.status(HttpStatus.OK).body(createsList) :
                ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
    }
}
