package com.example.firstproject1.api;

import com.example.firstproject1.annotation.RunningTime;
import com.example.firstproject1.dto.CommentDto;
import com.example.firstproject1.service.CommentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
public class CommentApiController {
    @Autowired
    private CommentService commentService;
//댓글 목록 조회
@GetMapping("/api/articles/{articleId}/comments")
public ResponseEntity<List<CommentDto>>  comment(@PathVariable Long articleId){
    // List<Comment> -> List<CommentDto> Comment(엔티티)를 DTO로 만들어서 반환,
    // 응답도 같이 보내주기 위해 ResponseEntity로 감싸기

    // 서비스에게 위임
    List<CommentDto> dtos = commentService.comments(articleId);

    // 결과 응답(성공한 경우만 있다고 가정)

    return ResponseEntity.status(HttpStatus.OK).body(dtos);
}

//댓글생성
@RunningTime
@PostMapping("/api/articles/{articleId}/comments")
public ResponseEntity<CommentDto> create(@PathVariable Long articleId,
                                         @RequestBody CommentDto dto) {
    // 서비스에게 위임
    CommentDto createdDto = commentService.create(articleId,dto);

    // 결과 응답( create 메소드안에서 에러를 발생시켜서 실패한경우에 응답코드를 자동으로 반환할 수 있도록 작성할 예정)
    return ResponseEntity.status(HttpStatus.OK).body(createdDto);
}
//댓글 수정
@PatchMapping("/api/comments/{id}")
public ResponseEntity<CommentDto> update(@PathVariable Long id,
                                         @RequestBody CommentDto dto){
    // 서비스에게 위임
    CommentDto updatedDto = commentService.update(id,dto);

   // 결과 응답(create메소드안에서 에러를 발생시켜서 실패한경우에 응답코드를
    // 자동으로 반환할 수 있도록 작성할 예정)
    return ResponseEntity.status(HttpStatus.OK).body(updatedDto);

}

//댓글 삭제

    @DeleteMapping("/api/comments/{id}")
    public ResponseEntity<CommentDto> delete(@PathVariable Long id){

        // 서비스에게 위임
        CommentDto deletedDto = commentService.delete(id);

        // 결과 응답
        return ResponseEntity.status(HttpStatus.OK).body(deletedDto);
    }

}
