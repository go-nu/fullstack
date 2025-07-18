package com.example.demo.controller;

import com.example.demo.dto.CouponDto;
import com.example.demo.service.CouponService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import java.util.List;

@Controller
@RequestMapping("/admin/coupons")
@RequiredArgsConstructor
public class CouponController {

    private final CouponService couponService;

    @GetMapping()
    public String couponPage(Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        model.addAttribute("loginUser", UserUtils.getUser(authentication));

        List<CouponDto> allCoupons = couponService.getAllCoupons();
        model.addAttribute("coupons", allCoupons);
        return "/admin/couponPage";
    }

    @PostMapping("/add")
    public String createCoupon(CouponDto dto) {
        couponService.create(dto);
        return "redirect:/admin/coupons";
    }

}
