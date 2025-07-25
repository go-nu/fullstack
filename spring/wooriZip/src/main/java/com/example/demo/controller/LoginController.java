package com.example.demo.controller;

import com.example.demo.oauth2.CustomOAuth2User;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.LoginService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
@RequiredArgsConstructor
public class LoginController {
    private final LoginService loginService;

    @GetMapping("/user/login")
    public String loginForm() {
        return "user/login";
    }

    @GetMapping("/user/logout")
    public String logout(){
        return "redirect:/logout";
    }
}