package com.springboot.controller;

import com.springboot.domain.Mem;
import com.springboot.domain.Member;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.WebDataBinder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;

import java.io.File;
import java.io.IOException;

@Controller
public class Ex01Controller {

//    @RequestMapping(value = "/exam01", method = RequestMethod.GET)
//    public String requestMethod() {
//        return "viewPage01";
//    }

    @GetMapping("exam01")
    public String requestMethod(@RequestParam("id") String userId,
                                @RequestParam("passwd") String userPw,
                                Model model) {
        model.addAttribute("data1", "@RequestParam 예제");
        model.addAttribute("data2", "요청 파라미터 id 값: " + userId +
                "<br/> 요청 파라미터 password 값: " + userPw);
        return "viewPage";
    }

    @GetMapping("/exam03/{id}")
    public String requestMethod(@PathVariable("id") String userId, Model model) {
        model.addAttribute("data1", "@PathVariable 예제");
        model.addAttribute("data2", "경로 변수 id 값: " + userId);
        return "viewPage03";
    }

    @GetMapping("/exam05/{id}")
    public String requestMethod05(@PathVariable("id") String userId,
                                   @MatrixVariable("passwd") String userPw,
                                   Model model) {
        model.addAttribute("data1", "@MatrixVariable 예제");
        model.addAttribute("data2", "경로 변수 id 값: " + userId +
                "<br/> 요청 matrix 변수 passwd 값: " + userPw);
        return "viewPage03";
    }

    @GetMapping("/member")
    public String showForm() {
        return "viewPage02";
    }

    @PostMapping("/member")
    public String submitForm(@ModelAttribute Member member, Model model) {
        model.addAttribute("member", member);
        return "viewPage021";
    }

    @InitBinder
    public void initBinder(WebDataBinder binder) {
        // 나타낼 항목 나열
        binder.setAllowedFields("id", "passwd", "city", "sex", "greetings");
        // 지울 항목 나열
        // binder.setDisallowedFields("hobby");
    }

    @GetMapping("/exam01/form")
    public String requestForm() {
        return "viewPage71";
    }

    @PostMapping("/exam01/form")
    public String submitForm(MultipartHttpServletRequest request, Model model){
        String name = request.getParameter("name");
        MultipartFile file = request.getFile("fileImage");
        String filename = file.getOriginalFilename();
        File saveFile = new File("c:\\upload\\" + name + "_" + filename );

        try {
            file.transferTo(saveFile);
            model.addAttribute("data1", "MultipartHttpServletRequest 예제" );
            model.addAttribute("data2", filename);
            model.addAttribute("data3", saveFile.getName() );
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "viewPage_process";
    }

    @GetMapping("/exam02/form")
    public String requestForm2(Mem mem) {
        return "viewPage72";
    }

    @PostMapping("/exam02/form")
    public String submitForm2(@ModelAttribute Mem mem, Model model) {
        String name = mem.getName();
        MultipartFile file = mem.getFileImage();
        String filename = file.getOriginalFilename();
        File saveFile = new File("c:\\upload\\" + name + "_" + filename);

        try {
            file.transferTo(saveFile);

            model.addAttribute("data1","@RequestParam 예졔" );
            model.addAttribute("data2",filename);
            model.addAttribute("data3", saveFile.getName() );
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "viewPage_process";
    }
}
