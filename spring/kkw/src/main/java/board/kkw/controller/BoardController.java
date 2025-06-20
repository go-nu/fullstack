package board.kkw.controller;

import board.kkw.domain.Board;
import board.kkw.domain.Member;
import board.kkw.dto.BoardDTO;
import board.kkw.service.BoardService;
import jakarta.servlet.http.HttpSession;
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
    public String writeView(HttpSession session, Model model) {
        Member loginUser = (Member) session.getAttribute("loginUser");
        model.addAttribute("loginUser", loginUser);
        return "/board/write";
    }


    @PostMapping("/write")
    public String write(@ModelAttribute BoardDTO dto, HttpSession session) throws IOException {
        Member writer = (Member) session.getAttribute("loginUser");
        boardService.create(dto, writer);
        return "redirect:/boards";
    }

    @GetMapping()
    public String list(Model model, HttpSession session){
        Member loginUser = (Member) session.getAttribute("loginUser");
        model.addAttribute("loginUser", loginUser);
        model.addAttribute("boards", boardService.list());
        return "/board/list";
    }

    @GetMapping("/{num}")
    public String boardView(@PathVariable Long num, Model model, HttpSession session) {
        Member writer = (Member) session.getAttribute("loginUser");
        model.addAttribute("board", boardService.findById(num));
        model.addAttribute("loginUser", writer);
        return "/board/detail";
    }


    @GetMapping("/update/{num}")
    public String updateForm(@PathVariable Long num, Model model) {
        model.addAttribute("board", boardService.findById(num));
        return "/board/update";
    }

    @PostMapping("/update/{num}")
    public String update(@PathVariable Long num, BoardDTO dto) throws IOException {
        Board board = boardService.findById(num);
        boardService.update(num, dto);
        return "redirect:/boards";
    }

    @GetMapping("/delete/{num}")
    public String delete(@PathVariable Long num) {
        boardService.delete(boardService.findById(num));
        return "redirect:/boards";
    }
}
