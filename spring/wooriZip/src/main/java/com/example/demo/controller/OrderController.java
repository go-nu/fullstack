package com.example.demo.controller;

import com.example.demo.dto.CartDto;
import com.example.demo.dto.OrderDto;
import com.example.demo.oauth2.CustomOAuth2User;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.OrderService;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.security.Principal;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
@RequestMapping("/order")
@RequiredArgsConstructor
public class OrderController {

    // 로그 수집
    private static final Logger log = LoggerFactory.getLogger(OrderController.class);
    private final OrderService orderService;

    @GetMapping
    public String order(Authentication authentication, Model model) {
        if (authentication != null && authentication.getPrincipal() instanceof CustomUserDetails userDetails) {
            OrderDto orderDto = orderService.getOrder(userDetails.getUser().getEmail());
            model.addAttribute("loginUser", userDetails.getUser());
            model.addAttribute("orderDto", orderDto);
        } else if (authentication != null && authentication.getPrincipal() instanceof CustomOAuth2User oauth2User) {
            OrderDto orderDto = orderService.getOrder(oauth2User.getUser().getEmail());
            model.addAttribute("loginUser", oauth2User.getUser());
            model.addAttribute("orderDto", orderDto);
        }
        return "order/orderPayment";
    }


    @PostMapping
    public String processOrder(@RequestParam("type") String orderType,
                               @RequestParam("cartItemIds") List<Long> cartItemIds,
                               Model model, Authentication authentication) {
        String email = "";
        if (authentication != null && authentication.getPrincipal() instanceof CustomUserDetails userDetails) {
            email = userDetails.getUser().getEmail();
        } else if (authentication != null && authentication.getPrincipal() instanceof CustomOAuth2User oauth2User) {
            email = oauth2User.getUser().getEmail();
        }

        if (email == null) {
            return "redirect:/login";
        }

        log.info("들어온 값 : {}", cartItemIds);
        log.info("들어온 타입 : {}", orderType);

        OrderDto orderDto;
        if (orderType.equals("cart")) {
            orderDto = orderService.createOrder(email, cartItemIds);
            model.addAttribute("orderDto", orderDto);
        }
        return "redirect:/order";
    }


    @PostMapping("/remove")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> removeOrderItem(@RequestBody Map<String, Object> payload, Authentication authentication) {
        Long orderId = Long.valueOf(payload.get("orderId").toString());
        Long orderItemId = Long.valueOf(payload.get("orderItemId").toString());

        String email = "";
        if (authentication != null && authentication.getPrincipal() instanceof CustomUserDetails userDetails) {
            email = userDetails.getUser().getEmail();
        } else if (authentication != null && authentication.getPrincipal() instanceof CustomOAuth2User oauth2User) {
            email = oauth2User.getUser().getEmail();
        }

        Map<String, Object> response = new HashMap<>();
        try {
            orderService.removeOrderItem(orderId, orderItemId, email);
            response.put("success", true);
        } catch (Exception e) {
            response.put("success", false);
            response.put("message", e.getMessage());
        }
        return ResponseEntity.ok(response);
    }


    @PostMapping("/updateQuantity")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> updateOrderItemQuantity(@RequestBody Map<String, Object> payload) {
        Long orderItemId = Long.valueOf(payload.get("orderItemId").toString());
        int newCount = Integer.parseInt(payload.get("count").toString());

        orderService.updateOrderItemQuantity(orderItemId, newCount);

        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        return ResponseEntity.ok(response);
    }


    // 주문확인
    @GetMapping("/history")
    public String history(@RequestParam(value = "page", defaultValue = "0") int page,
                          Model model, Authentication authentication) {
        if (page < 0) {
            page = 0;
        }

        String email = "";
        if (authentication != null && authentication.getPrincipal() instanceof CustomUserDetails userDetails) {
            email = userDetails.getUser().getEmail();
        } else if (authentication != null && authentication.getPrincipal() instanceof CustomOAuth2User oauth2User) {
            email = oauth2User.getUser().getEmail();
        }

        List<OrderDto> order = orderService.history(email);
        Page<OrderDto> orderPage = orderService.orderPage(order, page);
        log.info("orderPage = {}", orderPage);

        model.addAttribute("orders", orderPage.getContent());
        model.addAttribute("currentPage", page);
        model.addAttribute("totalPages", orderPage.getTotalPages());
        model.addAttribute("maxPage", 5);
        return "order/orderHistory";
    }


    // 바로구매 처리
    @PostMapping("/now")
    public String orderNow(@RequestParam Long productId, CartDto dto , Authentication authentication, Model model) {
        log.info("바로구매로 넘어온 값 : 상품아이디 : {} , dto {}", productId, dto.toString());

        String email = "";
        if (authentication != null && authentication.getPrincipal() instanceof CustomUserDetails userDetails) {
            email = userDetails.getUser().getEmail();
        } else if (authentication != null && authentication.getPrincipal() instanceof CustomOAuth2User oauth2User) {
            email = oauth2User.getUser().getEmail();
        }

        OrderDto orderDto = orderService.createOrderByNow(dto, productId, email);
        model.addAttribute("orderDto", orderDto);
        return "redirect:/order";
    }

    @GetMapping("/search")
    public String searchOrders(
            @RequestParam("startDate") @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam("endDate") @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate,
            Model model) {
        List<OrderDto> orders = orderService.findOrdersByDateRange(startDate, endDate);
        model.addAttribute("orders", orders);
        return "order/orderHistory";
    }
}
