package com.example.oauth2.service;


import com.example.oauth2.constant.Role;
import com.example.oauth2.entity.Users;
import com.example.oauth2.oauth.GoogleUserInfo;
import com.example.oauth2.oauth.KakaoUserInfo;
import com.example.oauth2.oauth.NaverUserInfo;
import com.example.oauth2.oauth.OAuth2UserInfo;
import com.example.oauth2.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.OAuth2Error;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class PrincipalOauth2UserService extends DefaultOAuth2UserService {
//  PrincipalOauth2UserService는 OAuth2 로그인 성공 후 사용자 정보를 처리하고, 사용자 객체로 매핑
    private final UserRepository userRepository;

    @Override
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {
        OAuth2User oAuth2User = super.loadUser(userRequest);
        OAuth2UserInfo oAuth2UserInfo = null;

        String provider = userRequest.getClientRegistration().getRegistrationId();
//      provide가 google, naver, kakao 중 무엇인지 판별
        if ("google".equals(provider)) {
            oAuth2UserInfo = new GoogleUserInfo(oAuth2User.getAttributes());
        } else if ("naver".equals(provider)) {
            oAuth2UserInfo = new NaverUserInfo((Map<String, Object>) oAuth2User.getAttributes().get("response"));
        } else if ("kakao".equals(provider)) { // 카카오 추가
            oAuth2UserInfo = new KakaoUserInfo(oAuth2User.getAttributes());
        }

        if (oAuth2UserInfo == null) {
            throw new OAuth2AuthenticationException(new OAuth2Error("invalid_user_info", "사용자 정보를 가져올 수 없습니다", null));
        }

        String providerId = oAuth2UserInfo.getProviderId();
        String email = oAuth2UserInfo.getEmail();
        if (email == null || email.isEmpty()) {
            email = provider + "_" + providerId + "@noemail.com"; // 이메일이 없을 경우 임시 이메일 생성
        }
        String socialId = provider + "_" + providerId;

        Optional<Users> optionalUsers = userRepository.findByEmail(email);
        Users users;
        // 새로운 소셜 로그인 사용자면 DB에 회원가입
        if (optionalUsers.isEmpty()) {
            // 새로운 사용자 생성
            users = Users.builder()
                    .userId(providerId)
                    .name(oAuth2UserInfo.getName() != null ? oAuth2UserInfo.getName() : "")
                    .email(email)
                    .phone("") // 기본값 설정
                    .pCode("") // 기본값 설정
                    .loadAddr("") // 기본값 설정
                    .lotAddr("") // 기본값 설정
                    .detailAddr("") // 기본값 설정
                    .extraAddr("") // 기본값 설정
                    .userPw("SOCIAL_LOGIN") // 소셜 로그인 사용자 비밀번호는 기본값 설정
                    .provider(provider)
                    .providerId(providerId)
                    .socialId(socialId)
                    .role(Role.USER) // 기본 역할을 USER로 설정
                    .build();
            userRepository.save(users);
        } else {
            // 기존 사용자 업데이트
            users = optionalUsers.get();
            users.setName(oAuth2UserInfo.getName() != null ? oAuth2UserInfo.getName() : users.getName());
            users.setProviderId(providerId);
            users.setSocialId(socialId);
            // 사용자가 직접 입력한 정보를 업데이트
            if (oAuth2UserInfo.getPhone() != null) users.setPhone(oAuth2UserInfo.getPhone());
            if (oAuth2UserInfo.getPCode() != null) users.setPCode(oAuth2UserInfo.getPCode());
            if (oAuth2UserInfo.getLoadAddr() != null) users.setLoadAddr(oAuth2UserInfo.getLoadAddr());
            if (oAuth2UserInfo.getLotAddr() != null) users.setLotAddr(oAuth2UserInfo.getLotAddr());
            if (oAuth2UserInfo.getDetailAddr() != null) users.setDetailAddr(oAuth2UserInfo.getDetailAddr());
            if (oAuth2UserInfo.getExtraAddr() != null) users.setExtraAddr(oAuth2UserInfo.getExtraAddr());
            userRepository.save(users);
        }

        Map<String, Object> attributes = new HashMap<>(oAuth2User.getAttributes());
        attributes.put("userId", users.getUserId());

        return new DefaultOAuth2User(
                Collections.singleton(new SimpleGrantedAuthority("ROLE_USER")),
                attributes,
                "userId");
    }
}
