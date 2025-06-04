package RestapiBoard.board1.controller;

import RestapiBoard.board1.service.BoardService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class BasicController {

    @Autowired
    private BoardService boardService;

    @GetMapping("/form")
    public String formPage() {
        return "form";
    }
    /*
    list1.html 짝꿍
    @GetMapping("/list")
    public String formPage1() {
        return "list";
    }
    */
    @GetMapping("/list")
    public String formPage1(Model model) {
        model.addAttribute("boards", boardService.findAll());
        return "list";
    }
    /*
    @GetMapping("/update/{id}")
    public String updatePage(@PathVariable Long id, Model model) {
        model.addAttribute("board", boardService.findById(id));
        return "update";
    }
    */
    @GetMapping("/update")
    public String updatePage(@RequestParam Long id, Model model) {
        model.addAttribute("board", boardService.findById(id));
        return "update";
    }

    @GetMapping("/detail/{id}")
    public String detailPage(@PathVariable Long id, Model model) {
        model.addAttribute("board", boardService.findById(id));
        return "detail";
    }
}
