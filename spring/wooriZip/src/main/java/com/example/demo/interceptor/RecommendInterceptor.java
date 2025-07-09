package com.example.demo.interceptor;

import com.example.demo.entity.Users;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.RecommendLogService;
import com.example.demo.utils.SpringContextUtils;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.util.StopWatch;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.ModelAndView;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.Period;
import java.time.format.DateTimeFormatter;

@Slf4j
public class RecommendInterceptor  implements HandlerInterceptor {

    ThreadLocal<StopWatch> stopWatchLocal = new ThreadLocal<>();

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        StopWatch stopWatch = new StopWatch(handler.toString());
        stopWatch.start(handler.toString());
        stopWatchLocal.set(stopWatch);

        String nickname = "";
        String gender = "";
        String ageGroup = "";
        int residence = 0;

        Object principal = SecurityContextHolder.getContext().getAuthentication().getPrincipal();

        if (principal instanceof CustomUserDetails userDetails) {
            Users user = userDetails.getUser();
            nickname = user.getNickname();
            gender = user.getGender();
            residence = user.getResidenceType();
            ageGroup = getAgeGroup(user.getBirth());
        }

        String url = getURLPath(request);
        String productId = extractProductId(url);

        // 상품 페이지일 때만 추천 로그 출력
        if (productId != null) {
            log.info("User: {} ({}, {}, {})", nickname, gender, ageGroup, residence);
            log.info("Action: VIEW");
            log.info("ProductID: {}", productId);
            log.info("Timestamp: {}", getCurrentTime());

            RecommendLogService recommendLogService = SpringContextUtils.getBean(RecommendLogService.class);

            recommendLogService.saveLog(
                    nickname,
                    gender,
                    ageGroup,
                    residence,
                    Long.parseLong(productId)
            );
        }

        return true;
    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) {
        StopWatch stopWatch = stopWatchLocal.get();
        if (stopWatch != null) {
            stopWatch.stop();
            stopWatchLocal.remove();
        }
    }

    private String getURLPath(HttpServletRequest request) {
        String query = request.getQueryString();
        return request.getRequestURI() + (query != null ? "?" + query : "");
    }

    private String getCurrentTime() {
        return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
    }

    private String getAgeGroup(LocalDate birth) {
        int age = Period.between(birth, LocalDate.now()).getYears();
        if (age < 20) return "10s";
        else if (age < 30) return "20s";
        else if (age < 40) return "30s";
        else if (age < 50) return "40s";
        else return "50s-";
    }

    private String extractProductId(String url) {
        // 예: /products/123 → 123
        if (url != null && url.matches("^/products/\\d+$")) {
            return url.substring(url.lastIndexOf("/") + 1);
        }
        return null;
    }

}