package board.kkw.controller;

import board.kkw.domain.Board;
import board.kkw.dto.BoardDTO;
import board.kkw.service.BoardService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;

@Controller
@RequestMapping("/boards")
public class BoardController {

    @Autowired
    private BoardService boardService;

    @GetMapping("/write")
    public String writeView() {
        return "write";
    }

    @PostMapping("/write")
    public String write(@ModelAttribute BoardDTO dto) throws IOException {
        boardService.create(dto);
        return "redirect:/boards";
    }

    @GetMapping()
    public String listView(Model model) {
        model.addAttribute("boards", boardService.list());
        return "list";
    }

    @GetMapping("/{num}")
    public String boardView(@PathVariable Long num, Model model) {
        model.addAttribute("board", boardService.findById(num));
        return "detail";
    }

    @GetMapping("/update/{num}")
    public String updateForm(@PathVariable Long num, Model model) {
        model.addAttribute("board", boardService.findById(num));
        return "update";
    }

    @PostMapping("/update/{num}")
    public String update(@PathVariable Long num, @ModelAttribute Board board) {
        boardService.update(boardService.findById(num));
        return "redirect:/boards";
    }

    @GetMapping("/delete/{num}")
    public String delete(@PathVariable Long num) {
        boardService.delete(boardService.findById(num));
        return "redirect:/boards";
    }
}
