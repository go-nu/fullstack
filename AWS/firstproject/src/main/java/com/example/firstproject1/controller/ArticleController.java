package com.example.firstproject1.controller;

import com.example.firstproject1.dto.ArticleForm;
import com.example.firstproject1.dto.CommentDto;
import com.example.firstproject1.entity.Article;
import com.example.firstproject1.repository.ArticleRepository;
import com.example.firstproject1.service.ArticleService;
import com.example.firstproject1.service.CommentService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.List;

@Controller
@Slf4j // 로깅을 위한 어노테이션
public class ArticleController {

    @Autowired
    private ArticleRepository articleRepository;
    @Autowired
    private CommentService commentService;


    @GetMapping("/articles/new")
    public String newArticleForm() {
        return "articles/new"; //실제연결될 mustache
    }
    @PostMapping("/articles/create")
    public String createArticle(ArticleForm form){
        //System.out.println(form.toString());
        log.info(form.toString());
        //Dto -> Entity변환
        Article article = form.toEntity();
        //System.out.println(article.toString());
        log.info(article.toString());
        //2. Repository 에게 Entity를 DB안에 저장하게 함
        Article saved= articleRepository.save(article);
        //System.out.println(saved.toString());
        log.info(saved.toString());
        return "redirect:/articles/"+ saved.getId();
    }

    @GetMapping("/articles/{id}")
    public String show(@PathVariable Long id, Model model) {
        log.info("id:" + id);


        Article articleEntity = articleRepository.findById(id).orElse(null);
        List<CommentDto> commentsDtos= commentService.comments(id);

        model.addAttribute("article", articleEntity);
        model.addAttribute("commentDtos",commentsDtos);
        // 3: 보여줄 페이지를 설정
        return "articles/show";
    }

    @GetMapping("/articles")
    public String index(Model model) {
        // 1: 모든 Article 가져오기
        //1번방법 List<Article> articleEntityList = (List<Article>)articleRepository.findAll();
        //2번방법 Iterable<Article> articleEntityList = articleRepository.findAll();
        List<Article> articleEntityList = articleRepository.findAll();

        // 2: 가져온 Article 묶음을 뷰로 전달
        model.addAttribute("articleList", articleEntityList);
        // 3: 뷰 페이지를 설정
        return "articles/index";
    }

    @GetMapping("/articles/{id}/edit")
    public String edit(@PathVariable Long id, Model model) {
        // 수정할 데이터 가져오기
        Article articleEntity = articleRepository.findById(id).orElse(null);

        // 모델에 데이터 등록
        model.addAttribute("article",articleEntity);

        return "articles/edit";
    }
    @PostMapping("/articles/update")
    public String update(ArticleForm form){
        log.info(form.toString());
        //1. dto를 엔티티로 변환
        Article articleEntity = form.toEntity();
        log.info(articleEntity.toString());

        //2. 엔티티를 db로 저장
        //2-1 db에서 기존의 데이터를 가져옴
        Article target = articleRepository.findById(articleEntity.getId()).orElse(null);

        //2-2 기존 데이터가 있다면, 값을 갱신

        if(target != null){
            articleRepository.save(articleEntity);
        }

        //3. 수정 결과 페이지로 리다이렉트

        return "redirect:/articles/"+articleEntity.getId() ;
    }
    @GetMapping("/articles/{id}/delete")
    public String delete(@PathVariable Long id, RedirectAttributes rttr) {
        log.info("삭제 요청");

        // 1: 삭제 대상을 가져온다.
        Article target = articleRepository.findById(id).orElse(null);

        // 2: 대상을 삭제한다.
        if(target != null) {
            articleRepository.delete(target);
            rttr.addFlashAttribute("msg", "삭제가 완료되었습니다.");
        }

        // 3: 결과 페이지로 리다이렉트한다.
        return "redirect:/articles";
    }


}
