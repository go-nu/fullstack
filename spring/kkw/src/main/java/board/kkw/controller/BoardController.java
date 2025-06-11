package board.kkw.controller;

import board.kkw.domain.Board;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class BoardController {

    @GetMapping("/write")
    public String write(Model model) {
        model.addAttribute("board", new Board());
        return "/write";
    }
}
