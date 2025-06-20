package board.kkw.controller;

import board.kkw.domain.Member;
import board.kkw.dto.MemberDto;
import board.kkw.service.MemberService;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/member")
public class MemberController {

    @Autowired
    private MemberService memberService;

    @GetMapping("/signUp")
    public String signinForm(){
        return "/member/signUp";
    }

    @PostMapping("/signUp")
    public String signUp(MemberDto dto){
        memberService.signUp(dto);
        return "redirect:/member/login";
    }

    @GetMapping("/login")
    public String loginForm(){
        return "/member/login";
    }

    @PostMapping("/login")
    public String login(MemberDto dto, HttpSession session){
        Member member = memberService.login(dto);
        if (member == null) {
            // 로그인 실패 처리 (예: 다시 로그인 페이지로 이동)
            return "redirect:/member/login?error";
        }
        session.setAttribute("loginUser", member);
        return "redirect:/boards";
    }

    @GetMapping("/logout")
    public String logout(HttpSession session){
        session.invalidate();
        return "redirect:/boards";
    }
}
