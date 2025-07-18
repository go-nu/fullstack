package com.example.demo.controller;

import com.example.demo.dto.*;
import com.example.demo.service.CartService;
import com.example.demo.service.OrderService;
import jakarta.servlet.http.HttpServletRequest;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

@Log4j2
@Controller
@RequiredArgsConstructor
public class PaymentController {

    private final OrderService orderService;
    private final CartService cartService;

    @PostMapping(value = "/confirm")
    public String confirmPayment(@RequestBody String jsonBody, Model model, Authentication authentication) {
        model.addAttribute("loginUser", UserUtils.getUser(authentication));

        log.info("confirmPayment called with body: {}", jsonBody);  // 요청이 들어오는지 로그 확인
        JSONParser parser = new JSONParser();
        String orderId;
        String amount;
        String paymentKey;
        String paymentMethod;
        String payInfo;

        try {
            JSONObject requestData = (JSONObject) parser.parse(jsonBody);
            paymentKey = (String) requestData.get("paymentKey");
            orderId = (String) requestData.get("orderId");
            amount = (String) requestData.get("amount");

            log.info("결제 정보: paymentKey={}, orderId={}, amount={}", paymentKey, orderId, amount);

            if (paymentKey == null || paymentKey.isEmpty() || orderId == null || orderId.isEmpty() || amount == null || amount.isEmpty()) {
                throw new IllegalArgumentException("결제 정보가 잘못되었습니다.");
            }

        } catch (ParseException e) {
            log.error("Error parsing JSON request body", e);
            return "redirect:/fail?message=Invalid JSON format&code=400";
        } catch (IllegalArgumentException e) {
            log.error("Invalid payment information", e);
            return "redirect:/fail?message=" + e.getMessage() + "&code=400";
        }

        JSONObject obj = new JSONObject();
        obj.put("orderId", orderId);
        obj.put("amount", amount);
        obj.put("paymentKey", paymentKey);

        String widgetSecretKey = "test_gsk_docs_OaPz8L5KdmQXkzRz3y47BMw6";
        Base64.Encoder encoder = Base64.getEncoder();
        byte[] encodedBytes = encoder.encode((widgetSecretKey + ":").getBytes(StandardCharsets.UTF_8));
        String authorizations = "Basic " + new String(encodedBytes);

        URL url;
        HttpURLConnection connection;
        try {
            url = new URL("https://api.tosspayments.com/v1/payments/confirm");
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestProperty("Authorization", authorizations);
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);

            try (OutputStream outputStream = connection.getOutputStream()) {
                outputStream.write(obj.toString().getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            log.error("Error connecting to payment API", e);
            return "redirect:/fail?message=Payment API connection failed&code=500";
        }

        int code;
        try {
            code = connection.getResponseCode();
        } catch (Exception e) {
            log.error("Error getting response from payment API", e);
            return "redirect:/fail?message=Payment API response failed&code=500";
        }

        boolean isSuccess = code == 200;
        try (InputStream responseStream = isSuccess ? connection.getInputStream() : connection.getErrorStream();
             Reader reader = new InputStreamReader(responseStream, StandardCharsets.UTF_8)) {

            JSONObject jsonObject = (JSONObject) parser.parse(reader);
            if (isSuccess) {
                JSONObject easyPayObject = (JSONObject) jsonObject.get("easyPay");
                paymentMethod = (String) easyPayObject.get("provider");
                payInfo = (String) jsonObject.get("method");

                log.info("성공 로그: paymentMethod={}, payInfo={}", paymentMethod, payInfo);

                // 검증시작
                if (orderService.validatePayment(orderId, Integer.parseInt(amount))) {
                    orderService.updateOrderWithPaymentInfo(orderId, paymentMethod, payInfo);
                    cartService.clearCart(orderId);
                    return "redirect:/success?orderId=" + orderId + "&amount=" + amount + "&paymentKey=" + paymentKey;
                } else {
                    log.error("Payment validation failed for orderId: {}", orderId);
                    orderService.failOrder(orderId); // 결제 검증 실패 시 주문을 실패 처리합니다.
                    return "redirect:/fail?message=Payment validation failed&code=400";
                }
            } else {
                String errorMessage = (String) jsonObject.get("message");
                String errorCode = (String) jsonObject.get("code");
                log.error("Payment API error: {} - {}", errorCode, errorMessage);
                orderService.failOrder(orderId);
                return "redirect:/fail?message=" + errorMessage + "&code=" + errorCode;
            }
        } catch (ParseException e) {
            log.error("Error parsing JSON response from payment API", e);
            return "redirect:/fail?message=Invalid JSON response&code=500";
        } catch (Exception e) {
            log.error("Unexpected error during payment confirmation", e);
            return "redirect:/fail?message=Unexpected error&code=500";
        }
    }

    @GetMapping(value = "/success")
    public String paymentRequest(HttpServletRequest request, Model model, Authentication authentication) {
        model.addAttribute("loginUser", UserUtils.getUser(authentication));
        String orderId = request.getParameter("orderId");
        OrderDto orderDto = orderService.getOrderByOrderId(orderId);
        model.addAttribute("orderDto", orderDto);
        return "order/orderComplete";
    }

    @PostMapping("/pay/checkout")
    @ResponseBody
    public ResponseEntity<Map<String, String>> initiatePayment(@RequestBody InitiatePaymentRequestDto dto, Authentication authentication) {
        String orderId = dto.getOrderId();
        String email = UserUtils.getEmail(authentication);

        try {
            OrderDto orderDto = orderService.getOrder(email);
            String amount = String.valueOf(orderDto.getTotalPrice());
            String paymentKey = orderDto.getOrderId();

            Map<String, String> response = new HashMap<>();
            response.put("success", "true");
            response.put("orderId", orderId);
            response.put("amount", amount);
            response.put("paymentKey", paymentKey);
            response.put("redirectUrl", "/pay/checkoutPage?orderId=" + orderId + "&amount=" + amount + "&paymentKey=" + paymentKey);

            HttpServletRequest request = ((ServletRequestAttributes) RequestContextHolder.currentRequestAttributes()).getRequest();
            request.getSession().setAttribute("orderDto", orderDto);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, String> response = new HashMap<>();
            response.put("success", "false");
            response.put("message", "결제 요청 중 오류가 발생했습니다.");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }

    @GetMapping(value = "/pay/checkoutPage")
    public String checkoutPage(@RequestParam String orderId, @RequestParam String amount,
                               @RequestParam String paymentKey, Model model) {
        HttpServletRequest request = ((ServletRequestAttributes) RequestContextHolder.currentRequestAttributes()).getRequest();
        OrderDto orderDto = (OrderDto) request.getSession().getAttribute("orderDto");

        model.addAttribute("orderId", orderId);
        model.addAttribute("amount", amount);
        model.addAttribute("paymentKey", paymentKey);
        model.addAttribute("orderDto", orderDto);
        return "pay/checkout";
    }

    @GetMapping(value = "/fail")
    public String failPayment(@RequestParam String code, @RequestParam String message, Model model) {
        model.addAttribute("code", code);
        model.addAttribute("message", message);
        return "pay/fail";
    }

    @PostMapping("/cancel")
    @ResponseBody
    public ResponseEntity<String> cancelOrder(@RequestBody CancelOrderRequestDto dto) {
        String orderId = dto.getOrderId();
        try {
            orderService.failOrder(orderId);
            return ResponseEntity.ok("주문 취소가 정상적으로 완료되었습니다");
        } catch (Exception e) {
            log.error("Error canceling order", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Failed to cancel order");
        }
    }
}
