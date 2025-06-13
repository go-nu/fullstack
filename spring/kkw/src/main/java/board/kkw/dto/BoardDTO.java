package board.kkw.dto;

import lombok.Getter;
import lombok.Setter;
import org.springframework.web.multipart.MultipartFile;

@Getter
@Setter
public class BoardDTO {
    private String title;
    private String writer;
    private String content;
    private MultipartFile img;
}
