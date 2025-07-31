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

        // 0. 자주하는질문 관련 키워드가 포함된 경우 (가장 먼저 체크)
        if (containsFaqKeywords(lowerMessage)) {
            return handleFaqInquiry(lowerMessage);
        }

        // 1. 카테고리 관련 키워드가 포함된 경우 (상품 검색보다 먼저 체크)
        if (containsCategoryKeywords(lowerMessage)) {
            return handleCategoryInquiry(lowerMessage);
        }

        // 2. 상품 검색 관련 키워드가 포함된 경우
        if (containsProductKeywords(lowerMessage)) {
            return handleProductSearch(lowerMessage);
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
        // 상품 검색으로 처리할 키워드들 (개별 가구명 + 새로운 카테고리 키워드 포함)
        String[] keywords = {"상품", "제품", "소파", "쇼파", "테이블", "식탁", "책상", "의자", "침대", "서랍", "수납장", "진열장", "책장", "선반", "행거", "붙박이장", "거울", "장식", "인테리어", "추천", "인기", "베스트",
                "조명", "스탠드", "천장등", "무드등", "패브릭", "커튼", "러그", "침구", "정리함", "옷걸이", "식기", "조리도구", "욕실용품", "청소용품", "액자", "시계", "디퓨저"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsCategoryKeywords(String message) {
        // 카테고리 관련 키워드들 (일반적인 분류 관련 단어 + 구체적인 가구명 + 새로운 카테고리)
        String[] keywords = {"카테고리", "분류", "종류", "가격대", "스타일", "컬러", "색상", "재질", "가구",
                "소파", "쇼파", "침대", "테이블", "식탁", "책상", "의자", "서랍", "수납장",
                "진열장", "책장", "선반", "행거", "붙박이장", "거울",
                "조명", "스탠드", "천장등", "무드등", "패브릭", "커튼", "러그", "침구", "정리함", "옷걸이", "식기", "조리도구", "욕실용품", "청소용품", "액자", "시계", "디퓨저"};
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
        String[] keywords = {"안녕", "하이", "hello", "hi", "반가워", "처음", "이전 질문 목록"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsEventKeywords(String message) {
        String[] keywords = {"이벤트", "프로모션", "세일", "특가", "행사", "기념일", "쿠폰"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private boolean containsFaqKeywords(String message) {
        String[] keywords = {"자주하는질문", "자주", "faq", "질문", "궁금", "문의"};
        return Arrays.stream(keywords).anyMatch(message::contains);
    }

    private ChatbotResponseDto handleProductSearch(String message) {
        List<Product> products = new ArrayList<>();
        String responseMessage = "";
        Long categoryId = null;

        // 각 가구 종류별로 상품 검색 (동적 카테고리 ID 사용)
        if (message.contains("소파") || message.contains("쇼파")) {
            // 소파 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("소파");
            responseMessage = "소파 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("소파");
        } else if (message.contains("테이블")) {
            // 테이블 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("테이블");
            responseMessage = "테이블 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("테이블.식탁.책상");
        } else if (message.contains("의자")) {
            // 의자 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("의자");
            responseMessage = "의자 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("의자");
        } else if (message.contains("침대")) {
            // 침대 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("침대");
            responseMessage = "침대 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("침대");
        } else if (message.contains("서랍") || message.contains("수납장")) {
            // 서랍/수납장 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("서랍");
            responseMessage = "서랍/수납장 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("서랍.수납장");
        } else if (message.contains("진열장") || message.contains("책장") || message.contains("선반")) {
            // 진열장/책장/선반 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("진열장");
            responseMessage = "진열장/책장/선반 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("진열장.책장.선반");
        } else if (message.contains("행거") || message.contains("붙박이장")) {
            // 행거/붙박이장 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("행거");
            responseMessage = "행거/붙박이장 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("행거.붙박이장");
        } else if (message.contains("거울")) {
            // 거울 관련 상품 검색
            products = productRepository.findByNameContainingIgnoreCase("거울");
            responseMessage = "거울 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("거울");
        }
        // 조명 세부 카테고리
        else if (message.contains("스탠드")) {
            products = productRepository.findByNameContainingIgnoreCase("스탠드");
            responseMessage = "스탠드 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("스탠드");
        } else if (message.contains("천장등")) {
            products = productRepository.findByNameContainingIgnoreCase("천장등");
            responseMessage = "천장등 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("천장등");
        } else if (message.contains("무드등")) {
            products = productRepository.findByNameContainingIgnoreCase("무드등");
            responseMessage = "무드등 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("무드등");
        } else if (message.contains("벽등")) {
            products = productRepository.findByNameContainingIgnoreCase("벽등");
            responseMessage = "벽등 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("벽등");
        }
        // 패브릭 세부 카테고리
        else if (message.contains("커튼")) {
            products = productRepository.findByNameContainingIgnoreCase("커튼");
            responseMessage = "커튼 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("커튼");
        } else if (message.contains("러그")) {
            products = productRepository.findByNameContainingIgnoreCase("러그");
            responseMessage = "러그 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("러그");
        } else if (message.contains("침구")) {
            products = productRepository.findByNameContainingIgnoreCase("침구");
            responseMessage = "침구 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("침구");
        } else if (message.contains("쿠션")) {
            products = productRepository.findByNameContainingIgnoreCase("쿠션");
            responseMessage = "쿠션 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("쿠션");
        }
        // 수납/정리 세부 카테고리
        else if (message.contains("정리함")) {
            products = productRepository.findByNameContainingIgnoreCase("정리함");
            responseMessage = "정리함 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("정리함");
        } else if (message.contains("옷걸이")) {
            products = productRepository.findByNameContainingIgnoreCase("옷걸이");
            responseMessage = "옷걸이 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("옷걸이");
        } else if (message.contains("수납용품")) {
            products = productRepository.findByNameContainingIgnoreCase("수납용품");
            responseMessage = "수납용품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("수납용품");
        } else if (message.contains("옷장용품")) {
            products = productRepository.findByNameContainingIgnoreCase("옷장용품");
            responseMessage = "옷장용품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("옷장용품");
        }
        // 주방용품 세부 카테고리
        else if (message.contains("식기")) {
            products = productRepository.findByNameContainingIgnoreCase("식기");
            responseMessage = "식기 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("식기");
        } else if (message.contains("조리도구")) {
            products = productRepository.findByNameContainingIgnoreCase("조리도구");
            responseMessage = "조리도구를 찾아드렸습니다.";
            categoryId = findCategoryIdByName("조리도구");
        } else if (message.contains("주방정리")) {
            products = productRepository.findByNameContainingIgnoreCase("주방정리");
            responseMessage = "주방정리 용품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("주방정리");
        } else if (message.contains("주방가전")) {
            products = productRepository.findByNameContainingIgnoreCase("주방가전");
            responseMessage = "주방가전을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("주방가전");
        }
        // 생활용품 세부 카테고리
        else if (message.contains("욕실용품")) {
            products = productRepository.findByNameContainingIgnoreCase("욕실용품");
            responseMessage = "욕실용품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("욕실용품");
        } else if (message.contains("청소용품")) {
            products = productRepository.findByNameContainingIgnoreCase("청소용품");
            responseMessage = "청소용품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("청소용품");
        } else if (message.contains("생활잡화")) {
            products = productRepository.findByNameContainingIgnoreCase("생활잡화");
            responseMessage = "생활잡화를 찾아드렸습니다.";
            categoryId = findCategoryIdByName("생활잡화");
        } else if (message.contains("계절용품")) {
            products = productRepository.findByNameContainingIgnoreCase("계절용품");
            responseMessage = "계절용품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("계절용품");
        }
        // 인테리어소품 세부 카테고리
        else if (message.contains("액자")) {
            products = productRepository.findByNameContainingIgnoreCase("액자");
            responseMessage = "액자 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("액자");
        } else if (message.contains("시계")) {
            products = productRepository.findByNameContainingIgnoreCase("시계");
            responseMessage = "시계 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("시계");
        } else if (message.contains("디퓨저")) {
            products = productRepository.findByNameContainingIgnoreCase("디퓨저");
            responseMessage = "디퓨저 상품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("디퓨저");
        } else if (message.contains("장식품")) {
            products = productRepository.findByNameContainingIgnoreCase("장식품");
            responseMessage = "장식품을 찾아드렸습니다.";
            categoryId = findCategoryIdByName("장식품");
        } else if (message.contains("인기") || message.contains("베스트") || message.contains("추천")) {
            // 인기 상품 (가격순으로 정렬하여 상위 5개)
            products = productRepository.findTop5ByOrderByPriceAsc();
            responseMessage = "인기 상품을 추천해드립니다.";
        } else {
            // 위 조건에 해당하지 않는 경우 - 최신 상품 5개 반환
            products = productRepository.findTop5ByOrderByCreatedAtDesc();
            responseMessage = "최신 상품을 보여드립니다.";
        }

        // 카테고리 ID가 있으면 해당 카테고리로 필터링된 링크 생성
        String link = categoryId != null ? "/products?category=" + categoryId : "/products";

        // 상품 목록 응답 생성
        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("product_list")  // 상품 목록 타입
                .products(products)    // 검색된 상품들
                .link(link)           // 카테고리 필터링된 상품 목록 페이지 링크
                .build();
    }

    private ChatbotResponseDto handleCategoryInquiry(String message) {
        // "카테고리 보여줘" 키워드가 포함된 경우 - 카테고리만 표시
        if (message.contains("카테고리")) {
            // 최상위 카테고리 목록 조회 (대분류)
            List<Category> categories = categoryRepository.findByParentIsNull();
            StringBuilder responseMessage = new StringBuilder("우리 쇼핑몰의 주요 카테고리입니다:<br><br>");

            // 각 카테고리명을 응답 메시지에 추가
            for (Category category : categories) {
                responseMessage.append("• ").append(category.getName()).append("\n");
            }

            // 카테고리 버튼들만 제공
            List<String> categorySuggestions = Arrays.asList(
                    "가구", "조명", "패브릭", "수납/정리", "주방용품", "생활용품", "인테리어소품"
            );

            // 카테고리 선택 응답 생성
            return ChatbotResponseDto.builder()
                    .message(responseMessage.toString())
                    .type("category_selection")  // 카테고리 선택 타입
                    .suggestions(categorySuggestions)  // 카테고리 버튼들만
                    .link("/products")  // 상품 목록 페이지 링크
                    .build();
        } else if (message.contains("가구")) {
            // "가구" 키워드가 포함된 경우 - 가구 하위 카테고리만 표시
            String responseMessage = "가구 카테고리의 세부 분류입니다:<br><br>" +
                    "• 침대 (침대프레임, 침대+매트리스, 침대부속가구)<br>" +
                    "• 테이블·식탁·책상 (식탁, 사무용책상, 좌식책상)<br>" +
                    "• 소파 (일반소파, 좌식소파, 리클라이너)<br>" +
                    "• 서랍·수납장 (서랍, 수납장, 협탁)<br>" +
                    "• 진열장·책장·선반 (진열장, 책장, 선반)<br>" +
                    "• 의자 (학생·사무용의자, 식탁의자, 스툴, 좌식의자)<br>" +
                    "• 행거·붙박이장 (행거, 붙박이장)<br>" +
                    "• 거울 (전신거울, 벽거울, 탁상거울)<br><br>" +
                    "원하시는 가구 종류를 말씀해 주세요!";

            List<String> furnitureSuggestions = Arrays.asList(
                    "침대", "테이블.식탁.책상", "소파", "서랍.수납장",
                    "진열장.책장.선반", "의자", "행거.붙박이장", "거울"
            );

            return ChatbotResponseDto.builder()
                    .message(responseMessage)
                    .type("furniture_subcategory")
                    .suggestions(furnitureSuggestions)
                    .build();
        } else if (message.contains("조명")) {
            // "조명" 키워드가 포함된 경우 - 조명 하위 카테고리만 표시
            String responseMessage = "조명 카테고리의 세부 분류입니다:<br><br>" +
                    "• 스탠드 (플로어스탠드, 테이블스탠드, LED스탠드)<br>" +
                    "• 천장등 (LED천장등, 다운라이트, 펜던트)<br>" +
                    "• 무드등 (LED무드등, 네온사인, 감성조명)<br>" +
                    "• 벽등 (벽걸이등, 실내벽등, 인테리어벽등)<br><br>" +
                    "원하시는 조명 종류를 말씀해 주세요!";

            List<String> lightingSuggestions = Arrays.asList(
                    "스탠드", "천장등", "무드등", "벽등"
            );

            return ChatbotResponseDto.builder()
                    .message(responseMessage)
                    .type("lighting_subcategory")
                    .suggestions(lightingSuggestions)
                    .build();
        } else if (message.contains("패브릭")) {
            // "패브릭" 키워드가 포함된 경우 - 패브릭 하위 카테고리만 표시
            String responseMessage = "패브릭 카테고리의 세부 분류입니다:<br><br>" +
                    "• 커튼 (일반커튼, 블라인드, 롤스크린)<br>" +
                    "• 러그 (카펫, 매트, 도어매트)<br>" +
                    "• 침구 (침대커버, 베개커버, 이불커버)<br>" +
                    "• 쿠션 (소파쿠션, 바닥쿠션, 인테리어쿠션)<br><br>" +
                    "원하시는 패브릭 종류를 말씀해 주세요!";

            List<String> fabricSuggestions = Arrays.asList(
                    "커튼", "러그", "침구", "쿠션"
            );

            return ChatbotResponseDto.builder()
                    .message(responseMessage)
                    .type("fabric_subcategory")
                    .suggestions(fabricSuggestions)
                    .build();
        } else if (message.contains("수납/정리")) {
            // "수납/정리" 키워드가 포함된 경우 - 수납/정리 하위 카테고리만 표시
            String responseMessage = "수납/정리 카테고리의 세부 분류입니다:<br><br>" +
                    "• 정리함 (플라스틱정리함, 종이정리함, 다용도정리함)<br>" +
                    "• 옷걸이 (옷걸이, 바지걸이, 넥타이걸이)<br>" +
                    "• 수납용품 (정리박스, 수납바구니, 정리대)<br>" +
                    "• 옷장용품 (옷장정리, 신발정리, 액세서리정리)<br><br>" +
                    "원하시는 수납/정리 용품을 말씀해 주세요!";

            List<String> storageSuggestions = Arrays.asList(
                    "정리함", "옷걸이", "수납용품", "옷장용품"
            );

            return ChatbotResponseDto.builder()
                    .message(responseMessage)
                    .type("storage_subcategory")
                    .suggestions(storageSuggestions)
                    .build();
        } else if (message.contains("주방용품")) {
            // "주방용품" 키워드가 포함된 경우 - 주방용품 하위 카테고리만 표시
            String responseMessage = "주방용품 카테고리의 세부 분류입니다:<br><br>" +
                    "• 식기 (그릇, 접시, 컵, 수저)<br>" +
                    "• 조리도구 (프라이팬, 냄비, 조리도구세트)<br>" +
                    "• 주방정리 (주방정리함, 조미료통, 주방용품)<br>" +
                    "• 주방가전 (전자레인지, 토스터기, 커피메이커)<br><br>" +
                    "원하시는 주방용품을 말씀해 주세요!";

            List<String> kitchenSuggestions = Arrays.asList(
                    "식기", "조리도구", "주방정리", "주방가전"
            );

            return ChatbotResponseDto.builder()
                    .message(responseMessage)
                    .type("kitchen_subcategory")
                    .suggestions(kitchenSuggestions)
                    .build();
        } else if (message.contains("생활용품")) {
            // "생활용품" 키워드가 포함된 경우 - 생활용품 하위 카테고리만 표시
            String responseMessage = "생활용품 카테고리의 세부 분류입니다:<br><br>" +
                    "• 욕실용품 (욕실용품, 샤워용품, 화장실용품)<br>" +
                    "• 청소용품 (청소도구, 빨래용품, 세제)<br>" +
                    "• 생활잡화 (생활용품, 실용품, 편의용품)<br>" +
                    "• 계절용품 (여름용품, 겨울용품, 계절잡화)<br><br>" +
                    "원하시는 생활용품을 말씀해 주세요!";

            List<String> dailySuggestions = Arrays.asList(
                    "욕실용품", "청소용품", "생활잡화", "계절용품"
            );

            return ChatbotResponseDto.builder()
                    .message(responseMessage)
                    .type("daily_subcategory")
                    .suggestions(dailySuggestions)
                    .build();
        } else if (message.contains("인테리어소품")) {
            // "인테리어소품" 키워드가 포함된 경우 - 인테리어소품 하위 카테고리만 표시
            String responseMessage = "인테리어소품 카테고리의 세부 분류입니다:<br><br>" +
                    "• 액자 (사진액자, 그림액자, 인테리어액자)<br>" +
                    "• 시계 (벽시계, 탁상시계, 인테리어시계)<br>" +
                    "• 디퓨저 (방향제, 디퓨저, 아로마용품)<br>" +
                    "• 장식품 (인테리어소품, 장식용품, 예술품)<br><br>" +
                    "원하시는 인테리어소품을 말씀해 주세요!";

            List<String> interiorSuggestions = Arrays.asList(
                    "액자", "시계", "디퓨저", "장식품"
            );

            return ChatbotResponseDto.builder()
                    .message(responseMessage)
                    .type("interior_subcategory")
                    .suggestions(interiorSuggestions)
                    .build();
        } else {
            // 특정 카테고리 요청인 경우 - 세부 카테고리 검색으로 처리
            return handleSubcategoryInquiry(message);
        }
    }

    private ChatbotResponseDto handleSubcategoryInquiry(String message) {
        String subcategoryType = "";      // 검색할 세부 카테고리 종류
        String responseMessage = "";    // 응답 메시지
        Long categoryId = null;         // 카테고리 ID

        // 가구 세부 카테고리
        if (message.contains("침대")) {
            subcategoryType = "침대";
            responseMessage = "침대 카테고리 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("침대");
        } else if (message.contains("테이블") || message.contains("식탁") || message.contains("책상")) {
            subcategoryType = "테이블·식탁·책상";
            responseMessage = "테이블/식탁/책상 카테고리 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("테이블.식탁.책상");
        } else if (message.contains("소파") || message.contains("쇼파")) {
            subcategoryType = "소파";
            responseMessage = "소파 카테고리 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("소파");
        } else if (message.contains("서랍") || message.contains("수납장")) {
            subcategoryType = "서랍·수납장";
            responseMessage = "서랍/수납장 카테고리 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("서랍.수납장");
        } else if (message.contains("진열장") || message.contains("책장") || message.contains("선반")) {
            subcategoryType = "진열장·책장·선반";
            responseMessage = "진열장/책장/선반 카테고리 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("진열장.책장.선반");
        } else if (message.contains("의자")) {
            subcategoryType = "의자";
            responseMessage = "의자 카테고리 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("의자");
        } else if (message.contains("행거") || message.contains("붙박이장")) {
            subcategoryType = "행거·붙박이장";
            responseMessage = "행거/붙박이장 카테고리 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("행거.붙박이장");
        } else if (message.contains("거울")) {
            subcategoryType = "거울";
            responseMessage = "거울 카테고리 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("거울");
        }
        // 조명 세부 카테고리
        else if (message.contains("스탠드")) {
            subcategoryType = "스탠드";
            responseMessage = "스탠드 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("스탠드");
        } else if (message.contains("천장등")) {
            subcategoryType = "천장등";
            responseMessage = "천장등 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("천장등");
        } else if (message.contains("무드등")) {
            subcategoryType = "무드등";
            responseMessage = "무드등 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("무드등");
        } else if (message.contains("벽등")) {
            subcategoryType = "벽등";
            responseMessage = "벽등 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("벽등");
        }
        // 패브릭 세부 카테고리
        else if (message.contains("커튼")) {
            subcategoryType = "커튼";
            responseMessage = "커튼 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("커튼");
        } else if (message.contains("러그")) {
            subcategoryType = "러그";
            responseMessage = "러그 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("러그");
        } else if (message.contains("침구")) {
            subcategoryType = "침구";
            responseMessage = "침구 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("침구");
        } else if (message.contains("쿠션")) {
            subcategoryType = "쿠션";
            responseMessage = "쿠션 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("쿠션");
        }
        // 수납/정리 세부 카테고리
        else if (message.contains("정리함")) {
            subcategoryType = "정리함";
            responseMessage = "정리함 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("정리함");
        } else if (message.contains("옷걸이")) {
            subcategoryType = "옷걸이";
            responseMessage = "옷걸이 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("옷걸이");
        } else if (message.contains("수납용품")) {
            subcategoryType = "수납용품";
            responseMessage = "수납용품을 보여드립니다.";
            categoryId = findCategoryIdByName("수납용품");
        } else if (message.contains("옷장용품")) {
            subcategoryType = "옷장용품";
            responseMessage = "옷장용품을 보여드립니다.";
            categoryId = findCategoryIdByName("옷장용품");
        }
        // 주방용품 세부 카테고리
        else if (message.contains("식기")) {
            subcategoryType = "식기";
            responseMessage = "식기 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("식기");
        } else if (message.contains("조리도구")) {
            subcategoryType = "조리도구";
            responseMessage = "조리도구를 보여드립니다.";
            categoryId = findCategoryIdByName("조리도구");
        } else if (message.contains("주방정리")) {
            subcategoryType = "주방정리";
            responseMessage = "주방정리 용품을 보여드립니다.";
            categoryId = findCategoryIdByName("주방정리");
        } else if (message.contains("주방가전")) {
            subcategoryType = "주방가전";
            responseMessage = "주방가전을 보여드립니다.";
            categoryId = findCategoryIdByName("주방가전");
        }
        // 생활용품 세부 카테고리
        else if (message.contains("욕실용품")) {
            subcategoryType = "욕실용품";
            responseMessage = "욕실용품을 보여드립니다.";
            categoryId = findCategoryIdByName("욕실용품");
        } else if (message.contains("청소용품")) {
            subcategoryType = "청소용품";
            responseMessage = "청소용품을 보여드립니다.";
            categoryId = findCategoryIdByName("청소용품");
        } else if (message.contains("생활잡화")) {
            subcategoryType = "생활잡화";
            responseMessage = "생활잡화를 보여드립니다.";
            categoryId = findCategoryIdByName("생활잡화");
        } else if (message.contains("계절용품")) {
            subcategoryType = "계절용품";
            responseMessage = "계절용품을 보여드립니다.";
            categoryId = findCategoryIdByName("계절용품");
        }
        // 인테리어소품 세부 카테고리
        else if (message.contains("액자")) {
            subcategoryType = "액자";
            responseMessage = "액자 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("액자");
        } else if (message.contains("시계")) {
            subcategoryType = "시계";
            responseMessage = "시계 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("시계");
        } else if (message.contains("디퓨저")) {
            subcategoryType = "디퓨저";
            responseMessage = "디퓨저 상품을 보여드립니다.";
            categoryId = findCategoryIdByName("디퓨저");
        } else if (message.contains("장식품")) {
            subcategoryType = "장식품";
            responseMessage = "장식품을 보여드립니다.";
            categoryId = findCategoryIdByName("장식품");
        } else {
            // 위 조건에 해당하지 않는 경우 - 일반 상품 검색으로 처리
            return handleProductSearch(message);
        }

        // 해당 세부 카테고리의 상품을 카테고리명으로 검색
        List<Product> products = productRepository.findByNameContainingIgnoreCase(subcategoryType);

        // 카테고리 ID가 있으면 해당 카테고리로 필터링된 링크 생성
        String link = categoryId != null ? "/products?category=" + categoryId : "/products";

        // 세부 카테고리별 상품 검색 결과 응답 생성
        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("product_list")  // 상품 목록 타입
                .products(products)    // 검색된 상품들
                .link(link)           // 카테고리 필터링된 상품 목록 페이지 링크
                .build();
    }

    private ChatbotResponseDto handleOrderInquiry(String message) {
        String responseMessage = "주문 및 배송 관련 안내입니다:<br><br>" +
                "📦 배송 안내<br>" +
                "• 배송 기간: 주문 후 2-3일 내 배송<br>" +
                "• 배송비: 50,000원 이상 구매 시 무료배송<br>" +
                "• 배송 지역: 전국 배송 가능<br><br>" +
                "📋 주문 확인<br>" +
                "• 주문 내역은 마이페이지에서 확인 가능<br>" +
                "• 배송 조회는 주문번호로 조회 가능";

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("text")  // 텍스트 타입
                .build();
    }

    private ChatbotResponseDto handleReturnInquiry(String message) {
        String responseMessage = "반품/교환 안내입니다:<br><br>" +
                "🔄 반품/교환 정책<br>" +
                "• 반품 기간: 배송 완료 후 7일 이내<br>" +
                "• 교환 기간: 배송 완료 후 14일 이내<br>" +
                "• 반품비: 고객 부담 (단순 변심의 경우)<br><br>" +
                "📞 문의<br>" +
                "• 고객센터: 1588-0000<br>" +
                "• 이메일: support@woorizip.com";

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("text")  // 텍스트 타입
                .build();
    }

    private ChatbotResponseDto handlePaymentInquiry(String message) {
        String responseMessage = "결제 안내입니다:<br><br>" +
                "💳 결제 방법<br>" +
                "• 신용카드, 체크카드<br>" +
                "• 무이자 할부: 3개월, 6개월, 12개월<br>" +
                "• 간편결제: 카카오페이, 네이버페이, 토스페이<br><br>" +
                "🎫 할인 혜택<br>" +
                "• 신규 가입 쿠폰: 10,000원 할인 쿠폰<br>" +
                "• 생일 쿠폰: 15% 할인<br>" +
                "• 포인트 적립: 구매 금액의 1%";

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("text")  // 텍스트 타입
                .build();
    }

    private ChatbotResponseDto handleEventInquiry(String message) {
        String responseMessage = "🎉 현재 진행 중인 이벤트입니다:<br><br>" +
                "🎊 신규 가입 이벤트<br>" +
                "• 가입 즉시 10,000원 할인 쿠폰 지급<br>" +
                "• 첫 구매 시 추가 5% 할인<br><br>" +
                "🎁 생일 축하 이벤트<br>" +
                "• 생일 월에 15% 할인 쿠폰 지급<br>" +
                "• 특별 선물 증정<br><br>" +
                "📅 시즌 이벤트<br>" +
                "• 봄맞이 가구 세일 (3월~4월)<br>" +
                "• 여름 특가 이벤트 (7월~8월)<br>" +
                "• 연말 감사제 (12월)";

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("text")  // 텍스트 타입
                .link("/event")  // 공지/이벤트 페이지 링크
                .build();
    }

    private ChatbotResponseDto handleFaqInquiry(String message) {
        String responseMessage = "❓ 자주하는질문 안내입니다:<br><br>" +
                "💬 채팅으로 물어보세요!<br><br>" +
                "📦 배송 관련<br>" +
                "• \"배송 안내 알려줘\" 또는 \"배송\" 입력<br><br>" +
                "💳 결제 관련<br>" +
                "• \"결제 방법 알려줘\" 또는 \"결제\" 입력<br><br>" +
                "🔄 반품/교환 관련<br>" +
                "• \"반품 정책 알려줘\" 또는 \"반품\" 입력<br><br>" +
                "🎉 이벤트 관련<br>" +
                "• \"이벤트 정보 알려줘\" 또는 \"이벤트\" 입력<br><br>" +
                "🛍️ 상품 검색<br>" +
                "• \"카테고리 보여줘\" - 가구, 조명, 패브릭 등<br>" +
                "• \"소파\", \"침대\", \"테이블\" 등 가구명 직접 입력<br>" +
                "• \"조명\", \"패브릭\", \"수납/정리\" 등 카테고리명 입력<br>" +
                "• \"스탠드\", \"커튼\", \"정리함\" 등 세부 상품명 입력<br><br>" +
                "💡 예시<br>" +
                "• \"배송비 얼마야?\"<br>" +
                "• \"소파 추천해줘\"<br>" +
                "• \"조명 카테고리 보여줘\"<br>" +
                "• \"무이자 할부 가능해?\"<br>" +
                "• \"반품 기간은?\"<br><br>" +
                "아래 버튼을 클릭하거나 직접 채팅으로 질문해보세요!";

        // 자주하는질문 세부 버튼들을 위한 제안사항 목록
        List<String> faqSuggestions = Arrays.asList(
                "배송 안내 알려줘",
                "카테고리 보여줘",
                "결제 방법 알려줘", 
                "반품 정책 알려줘",
                "이벤트 정보 알려줘"
        );

        return ChatbotResponseDto.builder()
                .message(responseMessage)
                .type("faq")  // FAQ 타입
                .suggestions(faqSuggestions)  // FAQ 세부 버튼들
                .build();
    }

    public List<String> getCommonSuggestions() {
        return Arrays.asList(
                "자주하는질문",
                "카테고리 보여줘",
                "배송 안내 알려줘",
                "반품 정책 알려줘",
                "결제 방법 알려줘",
                "이벤트 정보 알려줘"
        );
    }

    // 카테고리명으로 카테고리 ID를 찾는 메서드
    private Long findCategoryIdByName(String categoryName) {
        java.util.Optional<Category> category = categoryRepository.findByName(categoryName);
        if (category.isPresent()) {
            System.out.println("카테고리 찾음: " + categoryName + " -> ID: " + category.get().getId());
            return category.get().getId();
        } else {
            System.out.println("카테고리를 찾을 수 없음: " + categoryName);
            // 모든 카테고리 출력해서 디버깅
            System.out.println("=== 모든 카테고리 목록 ===");
            categoryRepository.findAll().forEach(cat ->
                    System.out.println("ID: " + cat.getId() + ", Name: " + cat.getName() + ", Depth: " + cat.getDepth())
            );
            System.out.println("========================");

            // 카테고리명에 점이 포함된 경우 점을 제거해서 다시 시도
            if (categoryName.contains(".")) {
                String nameWithoutDot = categoryName.replace(".", "");
                System.out.println("점 제거 후 다시 시도: " + nameWithoutDot);
                java.util.Optional<Category> categoryWithoutDot = categoryRepository.findByName(nameWithoutDot);
                if (categoryWithoutDot.isPresent()) {
                    System.out.println("카테고리 찾음 (점 제거 후): " + nameWithoutDot + " -> ID: " + categoryWithoutDot.get().getId());
                    return categoryWithoutDot.get().getId();
                }
            }

            return null;
        }
    }
} 