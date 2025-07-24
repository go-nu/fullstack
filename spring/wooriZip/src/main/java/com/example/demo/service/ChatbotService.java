package com.example.demo.service;

import com.example.demo.dto.ChatbotResponseDto;
import com.example.demo.entity.Category;
import com.example.demo.entity.Product;
import com.example.demo.repository.CategoryRepository;
import com.example.demo.repository.ProductRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ChatbotService {

    private final ProductRepository productRepository;
    private final CategoryRepository categoryRepository;

    public ChatbotResponseDto processMessage(String message) {
        // 메시지를 소문자로 변환하고 앞뒤 공백 제거
        String lowerMessage = message.toLowerCase().trim();

        // 1. 상품 검색 관련 키워드가 포함된 경우
        if (containsProductKeywords(lowerMessage)) {
            return handleProductSearch(lowerMessage);
        }

        // 2. 카테고리 관련 키워드가 포함된 경우
        if (containsCategoryKeywords(lowerMessage)) {
            return handleCategoryInquiry(lowerMessage);
        }

        // 3. 주문/배송 관련 키워드가 포함된 경우
        if (containsOrderKeywords(lowerMessage)) {
            return handleOrderInquiry(lowerMessage);
        }

        // 4. 반품/교환 관련 키워드가 포함된 경우
        if (containsReturnKeywords(lowerMessage)) {
            return handleReturnInquiry(lowerMessage);
        }

        // 5. 결제 관련 키워드가 포함된 경우
        if (containsPaymentKeywords(lowerMessage)) {
            return handlePaymentInquiry(lowerMessage);
        }

        // 6. 이벤트 관련 키워드가 포함된 경우
        if (containsEventKeywords(lowerMessage)) {
            return handleEventInquiry(lowerMessage);
        }

        // 7. 인사말 관련 키워드가 포함된 경우
        if (containsGreetingKeywords(lowerMessage)) {
            return ChatbotResponseDto.builder()
                    .message("안녕하세요! 우리집 쇼핑몰 챗봇입니다. 무엇을 도와드릴까요?")
                    .type("text")
                    .suggestions(getCommonSuggestions())
                    .build();
        }

        // 8. 위의 모든 조건에 해당하지 않는 경우 - 기본 응답
        return ChatbotResponseDto.builder()
                .message("죄송합니다. 질문을 이해하지 못했습니다. 다른 방법으로 질문해 주시거나 아래 제안사항 중 하나를 선택해 주세요.")
                .type("text")
                .suggestions(getCommonSuggestions())
                .build();
    }

    private boolean containsProductKeywords(String message) {
        // 상품 검색으로 처리할 키워드들 (개별 가구명 포함)
        String[] keywords = {"상품", "제품", "소파", "쇼파", "테이블", "식탁", "책상", "의자", "침대", "서랍", "수납장", "진열장", "책장", "선반", "행거", "붙박이장", "거울", "장식", "인테리어", "추천", "인기", "베스트"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsCategoryKeywords(String message) {
        // 카테고리 관련 키워드들 (일반적인 분류 관련 단어)
        String[] keywords = {"카테고리", "분류", "종류", "가격대", "스타일", "컬러", "색상", "재질", "가구"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsOrderKeywords(String message) {
        String[] keywords = {"주문", "배송", "배달", "택배", "언제", "도착", "배송비", "무료배송"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsReturnKeywords(String message) {
        String[] keywords = {"반품", "교환", "환불", "취소", "사이즈", "색상", "불만"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsPaymentKeywords(String message) {
        String[] keywords = {"결제", "카드", "무이자", "할부", "쿠폰", "할인", "포인트", "세일", "프로모션"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsGreetingKeywords(String message) {
        String[] keywords = {"안녕", "하이", "hello", "hi", "반가워", "처음"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsEventKeywords(String message) {
        String[] keywords = {"이벤트", "프로모션", "세일", "특가", "행사", "기념일", "쿠폰"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private ChatbotResponseDto handleProductSearch(String message) {
        List<Product> products = new ArrayList<>();
        String responseMessage = "";

        // 각 가구 종류별로 상품 검색
        if (message.contains("소파") || message.contains("쇼파")) {
            // 소파 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("소파");
            responseMessage = "소파 상품을 찾아드렸습니다.";
        } else if (message.contains("테이블")) {
            // 테이블 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("테이블");
            responseMessage = "테이블 상품을 찾아드렸습니다.";
        } else if (message.contains("의자")) {
            // 의자 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("의자");
            responseMessage = "의자 상품을 찾아드렸습니다.";
        } else if (message.contains("침대")) {
            // 침대 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("침대");
            responseMessage = "침대 상품을 찾아드렸습니다.";
        } else if (message.contains("서랍") || message.contains("수납장")) {
            // 서랍/수납장 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("서랍");
            responseMessage = "서랍/수납장 상품을 찾아드렸습니다.";
        } else if (message.contains("진열장") || message.contains("책장") || message.contains("선반")) {
            // 진열장/책장/선반 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("진열장");
            responseMessage = "진열장/책장/선반 상품을 찾아드렸습니다.";
        } else if (message.contains("행거") || message.contains("붙박이장")) {
            // 행거/붙박이장 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("행거");
            responseMessage = "행거/붙박이장 상품을 찾아드렸습니다.";
        } else if (message.contains("거울")) {
            // 거울 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("거울");
            responseMessage = "거울 상품을 찾아드렸습니다.";
        } else if (message.contains("인기") || message.contains("베스트") || message.contains("추천")) {
            // 인기 상품 (가격순으로 정렬하여 상위 5개)
            products = productRepository.findTop5ByOrderByPriceAsc();
            responseMessage = "인기 상품을 추천해드립니다.";
        } else {
            // 위 조건에 해당하지 않는 경우 - 최신 상품 5개 반환
            products = productRepository.findTop5ByOrderByCreatedAtDesc();
            responseMessage = "최신 상품을 보여드립니다.";
        }

        // 상품 목록 응답 생성
        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("product_list")  // 상품 목록 타입
                .products(products)    // 검색된 상품들
                .link("/products")     // 상품 목록 페이지 링크
                .build();
    }

    private ChatbotResponseDto handleCategoryInquiry(String message) {
        // "가구" 또는 "카테고리" 키워드가 포함된 경우
        if (message.contains("가구") || message.contains("카테고리")) {
            // 최상위 카테고리 목록 조회 (대분류)
            List<Category> categories = categoryRepository.findByParentIsNull();
            StringBuilder responseMessage = new StringBuilder("우리 쇼핑몰의 주요 카테고리입니다:\n");

            // 각 카테고리명을 응답 메시지에 추가
            for (Category category : categories) {
                responseMessage.append("• ").append(category.getName()).append("\n");
            }

            // 가구 하위 카테고리 버튼들을 위한 제안사항 목록
            List<String> furnitureSuggestions = Arrays.asList(
                    "침대", "테이블.식탁.책상", "소파", "서랍.수납장",
                    "진열장.책장.선반", "의자", "행거.붙박이장", "거울"
            );

            // 카테고리 선택 응답 생성
            return ChatbotResponseDto.builder()
                    .message(responseMessage.toString())
                    .type("category_selection")  // 카테고리 선택 타입
                    .suggestions(furnitureSuggestions)  // 가구 하위 카테고리 버튼들
                    .link("/products")  // 상품 목록 페이지 링크
                    .build();
        } else {
            // 특정 가구 종류 요청인 경우 - 가구별 상품 검색으로 처리
            return handleFurnitureTypeInquiry(message);
        }
    }

    private ChatbotResponseDto handleFurnitureTypeInquiry(String message) {
        String furnitureType = "";      // 검색할 가구 종류
        String responseMessage = "";    // 응답 메시지

        // 각 가구 종류별로 카테고리명 매칭
        if (message.contains("침대")) {
            furnitureType = "침대";
            responseMessage = "침대 카테고리 상품을 보여드립니다.";
        } else if (message.contains("테이블") || message.contains("식탁") || message.contains("책상")) {
            // 테이블, 식탁, 책상 중 하나라도 포함되면 테이블.식탁.책상 카테고리로 검색
            furnitureType = "테이블.식탁.책상";
            responseMessage = "테이블/식탁/책상 카테고리 상품을 보여드립니다.";
        } else if (message.contains("소파") || message.contains("쇼파")) {
            // 소파 또는 쇼파 중 하나라도 포함되면 소파 카테고리로 검색
            furnitureType = "소파";
            responseMessage = "소파 카테고리 상품을 보여드립니다.";
        } else if (message.contains("서랍") || message.contains("수납장")) {
            // 서랍 또는 수납장 중 하나라도 포함되면 서랍.수납장 카테고리로 검색
            furnitureType = "서랍.수납장";
            responseMessage = "서랍/수납장 카테고리 상품을 보여드립니다.";
        } else if (message.contains("진열장") || message.contains("책장") || message.contains("선반")) {
            // 진열장, 책장, 선반 중 하나라도 포함되면 진열장.책장.선반 카테고리로 검색
            furnitureType = "진열장.책장.선반";
            responseMessage = "진열장/책장/선반 카테고리 상품을 보여드립니다.";
        } else if (message.contains("의자")) {
            furnitureType = "의자";
            responseMessage = "의자 카테고리 상품을 보여드립니다.";
        } else if (message.contains("행거") || message.contains("붙박이장")) {
            // 행거 또는 붙박이장 중 하나라도 포함되면 행거.붙박이장 카테고리로 검색
            furnitureType = "행거.붙박이장";
            responseMessage = "행거/붙박이장 카테고리 상품을 보여드립니다.";
        } else if (message.contains("거울")) {
            furnitureType = "거울";
            responseMessage = "거울 카테고리 상품을 보여드립니다.";
        } else {
            // 위 조건에 해당하지 않는 경우 - 일반 상품 검색으로 처리
            return handleProductSearch(message);
        }

        // 해당 가구 종류의 상품을 카테고리명으로 검색
        List<Product> products = productRepository.findByNameContainingIgnoreCase(furnitureType);

        // 가구별 상품 검색 결과 응답 생성
        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("product_list")  // 상품 목록 타입
                .products(products)    // 검색된 상품들
                .link("/products")     // 상품 목록 페이지 링크
                .build();
    }

    private ChatbotResponseDto handleOrderInquiry(String message) {
        String responseMessage = "주문 및 배송 관련 안내입니다:\n\n" +
                "📦 배송 안내\n" +
                "• 배송 기간: 주문 후 2-3일 내 배송\n" +
                "• 배송비: 50,000원 이상 구매 시 무료배송\n" +
                "• 배송 지역: 전국 배송 가능\n\n" +
                "📋 주문 확인\n" +
                "• 주문 내역은 마이페이지에서 확인 가능\n" +
                "• 배송 조회는 주문번호로 조회 가능";

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("text")  // 텍스트 타입
                .link("/order/history")  // 주문 내역 페이지 링크
                .build();
    }

    private ChatbotResponseDto handleReturnInquiry(String message) {
        String responseMessage = "반품/교환 안내입니다:\n\n" +
                "🔄 반품/교환 정책\n" +
                "• 반품 기간: 배송 완료 후 7일 이내\n" +
                "• 교환 기간: 배송 완료 후 14일 이내\n" +
                "• 반품비: 고객 부담 (단순 변심의 경우)\n\n" +
                "📞 문의\n" +
                "• 고객센터: 1588-0000\n" +
                "• 이메일: support@woorizip.com";

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("text")  // 텍스트 타입
                .build();
    }

    private ChatbotResponseDto handlePaymentInquiry(String message) {
        String responseMessage = "결제 안내입니다:\n\n" +
                "💳 결제 방법\n" +
                "• 신용카드, 체크카드\n" +
                "• 무이자 할부: 3개월, 6개월, 12개월\n" +
                "• 간편결제: 카카오페이, 네이버페이\n\n" +
                "🎫 할인 혜택\n" +
                "• 신규 가입 쿠폰: 10% 할인\n" +
                "• 생일 쿠폰: 15% 할인\n" +
                "• 포인트 적립: 구매 금액의 1%";

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("text")  // 텍스트 타입
                .link("/user/mypage")  // 마이페이지 링크
                .build();
    }

    private ChatbotResponseDto handleEventInquiry(String message) {
        String responseMessage = "🎉 현재 진행 중인 이벤트입니다:\n\n" +
                "🎊 신규 가입 이벤트\n" +
                "• 가입 즉시 10,000원 할인 쿠폰 지급\n" +
                "• 첫 구매 시 추가 5% 할인\n\n" +
                "🎁 생일 축하 이벤트\n" +
                "• 생일 월에 15% 할인 쿠폰 지급\n" +
                "• 특별 선물 증정\n\n" +
                "📅 시즌 이벤트\n" +
                "• 봄맞이 가구 세일 (3월~4월)\n" +
                "• 여름 특가 이벤트 (7월~8월)\n" +
                "• 연말 감사제 (12월)";

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("text")  // 텍스트 타입
                .link("/event")  // 이벤트 페이지 링크
                .build();
    }

    public List<String> getCommonSuggestions() {
        return Arrays.asList(
                "가구 카테고리 보여줘",
                "배송 안내 알려줘",
                "반품 정책 알려줘",
                "결제 방법 알려줘",
                "이벤트 정보 알려줘"
        );
    }
} 