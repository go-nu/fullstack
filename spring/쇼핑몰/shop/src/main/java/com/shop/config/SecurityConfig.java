package com.shop.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.util.matcher.AntPathRequestMatcher;

@Configuration
@EnableWebSecurity
public class SecurityConfig {
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                // 로그인 설정
                .formLogin(form -> form
                        .loginPage("/members/login")
                        .defaultSuccessUrl("/")
                        .usernameParameter("email")
                        .failureUrl("/members/login/error")
                        .permitAll()
                )
                // 로그아웃 설정
                .logout(logout -> logout
                        .logoutRequestMatcher(new AntPathRequestMatcher("/members/logout"))
                        .logoutSuccessUrl("/")
                        .permitAll()
                )
                // 권한 설정: authorizeHttpRequests만 사용!
                .authorizeHttpRequests(authorize -> authorize
                        // 정적 자원 접근 허용
                        .requestMatchers("/css/**", "/js/**", "/img/**", "/images/**").permitAll()
                        // 회원, 상품, 홈 등 공용 페이지
                        .requestMatchers("/", "/members/**", "/item/**").permitAll()
                        // 관리자 페이지
                        .requestMatchers("/admin/**").hasRole("ADMIN")
                        // 그 외 모든 요청은 인증 필요
                        .anyRequest().authenticated()
                )
//                // 예외처리: 로그인 필요 시 커스텀 핸들러
//                .exceptionHandling(ex -> ex
//                        .authenticationEntryPoint(new CustomAuthenticationEntryPoint())
//                )
                // CSRF, CORS 등 기타 설정 유지
                .csrf().disable();

        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder(){
        return new BCryptPasswordEncoder();
    }
}
