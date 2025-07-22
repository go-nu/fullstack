package com.example.demo.controller;

import com.example.demo.dto.*;
import com.example.demo.entity.*;
import com.example.demo.repository.AttributeRepository;
import com.example.demo.repository.AttributeValueRepository;
import com.example.demo.repository.CategoryRepository;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.http.MediaType;

import java.util.*;

@Controller
@RequiredArgsConstructor
@Slf4j
public class ProductController {

    private final ProductService productService;
    private final WishlistService wishlistService;
    private final ReviewPostService reviewPostService;
    private final QnaPostService qnaPostService;
    private final ProductDetailService productDetailService;
    private final CategoryRepository categoryRepository;
    private final AttributeRepository attributeRepository;
    private final AttributeValueRepository attributeValueRepository;

    @GetMapping("/admin/products")
    public String showProductForm(Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        model.addAttribute("loginUser", UserUtils.getUser(authentication));

        ProductForm productForm = new ProductForm();

        // 기본적으로 하나의 모델을 추가
        ProductModelDto defaultModel = new ProductModelDto();
        productForm.getProductModelDtoList().add(defaultModel);  // 기본 모델 추가

        model.addAttribute("productForm", productForm);  // 상품 폼 전달
        // 속성/속성값 목록 추가
        model.addAttribute("attributes", attributeRepository.findAll());
        // AttributeValue를 DTO로 변환하여 전달
        List<AttributeValue> attributeValues = attributeValueRepository.findAllWithAttribute();
        List<AttributeValueDto> attributeValueDtos = attributeValues.stream()
                .map(av -> new AttributeValueDto(av.getId(), av.getValue(), av.getAttribute().getName()))
                .toList();
        model.addAttribute("attributeValues", attributeValueDtos);
        return "product/products";  // 상품 등록 페이지로 리턴
    }

    // 상품등록
    @PostMapping("/admin/products")
    public String createProduct(
            @ModelAttribute ProductForm form,
            @RequestParam("images") MultipartFile[] images,
            @RequestParam(value = "productModelDtoListJson", required = false) String modelsJson,
            Authentication authentication,
            Model model) {
        String email = UserUtils.getEmail(authentication);
        try {
            if (email == null) return "redirect:/user/login";

            Users loginUser = (Users) UserUtils.getUser(authentication);

            // 옵션 리스트 JSON 파싱 (프론트에서 넘어온 경우)
            if (modelsJson != null && !modelsJson.isEmpty()) {
                com.fasterxml.jackson.databind.ObjectMapper objectMapper = new com.fasterxml.jackson.databind.ObjectMapper();
                java.util.List<com.example.demo.dto.ProductModelDto> modelDtoList = objectMapper.readValue(
                        modelsJson, new com.fasterxml.jackson.core.type.TypeReference<java.util.List<com.example.demo.dto.ProductModelDto>>() {}
                );
                form.setProductModelDtoList(modelDtoList);
            }

            // 상품 등록 처리
            Long productId = productService.createProduct(form, java.util.Arrays.asList(images), loginUser);

            // 모델에 등록된 상품 정보를 전달
            model.addAttribute("productForm", form);  // 상품 등록 폼을 뷰로 전달

        } catch (Exception e) {
            e.printStackTrace();
            return "redirect:/error";
        }
        return "redirect:/products";  // 상품 목록 페이지로 리다이렉트
    }

    @GetMapping("/products")
    public String showProductList(@RequestParam(name = "category", required = false) Long categoryId,
                                  Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        model.addAttribute("loginUser", UserUtils.getUser(authentication));
        List<Product> productList = productService.findProducts(categoryId);
        model.addAttribute("products", productList);
        return "product/list"; // 실제 Thymeleaf 템플릿 경로에 맞게 조정
    }

    // 상품상세
    @GetMapping("/products/{id}")
    public String viewProduct(@PathVariable Long id,
                              @RequestParam(defaultValue = "1") int page,
                              @RequestParam(defaultValue = "latest") String sort,
                              @RequestParam(name = "qnaPage", defaultValue = "1") int qnaPage,
                              @RequestParam(name = "qnaFilter", defaultValue = "all") String qnaFilter,
                              Model model,
                              Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        Users user = email != null ? (Users) UserUtils.getUser(authentication) : null;

        // ───────────── 상품 상세 정보 ─────────────
        ProductDetailDto dto = productService.getProductDetail(id, user);
        model.addAttribute("product", dto);
        model.addAttribute("loginUser", user);

        // ───────────── 상품 상세정보 (이미지/규격) ─────────────
        com.example.demo.dto.ProductDetailInfoDto productDetail = productDetailService.findByProductId(id);
        model.addAttribute("productDetail", productDetail);

        // AttributeValue를 DTO로 변환하여 전달
        List<AttributeValue> attributeValues = attributeValueRepository.findAllWithAttribute();
        List<AttributeValueDto> attributeValueDtos = attributeValues.stream()
                .map(av -> new AttributeValueDto(av.getId(), av.getValue(), av.getAttribute().getName()))
                .toList();
        model.addAttribute("attributeValues", attributeValueDtos);

        // ───────────── 리뷰 데이터 ─────────────
        int reviewPageNum = Math.max(page - 1, 0);
        if (!sort.equals("latest") && !sort.equals("rating")) {
            sort = "latest";
        }

        Page<ReviewPost> reviewPage = reviewPostService.getReviews(id, reviewPageNum, sort);
        Map<Integer, Long> ratingSummary = reviewPostService.getRatingDistribution(id);
        Double averageRating = reviewPostService.getAverageRating(id);
        boolean hasWritten = (user != null && user.getEmail() != null)
                && reviewPostService.hasWrittenReview(id, user.getEmail());

        // ReviewPost를 ReviewPostDto로 변환
        Page<ReviewPostDto> reviewDtoPage = reviewPage.map(ReviewPostDto::fromEntity);

        model.addAttribute("reviewPage", reviewDtoPage != null ? reviewDtoPage : Page.empty());
        model.addAttribute("ratingSummary", ratingSummary != null ? ratingSummary : new HashMap<>());
        model.addAttribute("averageRating", averageRating != null ? averageRating : 0.0);
        model.addAttribute("hasWritten", hasWritten);
        model.addAttribute("sort", sort);
        model.addAttribute("productId", id);

        // ───────────── QnA 데이터 ─────────────
        int qnaPageNum = Math.max(qnaPage - 1, 0);
        List<QnaPostDto> qnaList = qnaPostService.getQnaPostDtoList(id, qnaPageNum, qnaFilter);
        long qnaTotal = qnaPostService.countByProduct(id, qnaFilter);

        model.addAttribute("qnaList", qnaList != null ? qnaList : new ArrayList<>());
        model.addAttribute("qnaTotal", qnaTotal);
        model.addAttribute("qnaPage", qnaPage);
        model.addAttribute("qnaFilter", qnaFilter);

        return "product/detail";
    }

    @PostMapping("/wishlist/toggle")
    public String toggleWishlist(@RequestParam Long productId,
                                 Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users user = (Users) UserUtils.getUser(authentication);
        wishlistService.toggleWishlist(user, productId);
        return "redirect:/products/" + productId;
    }

    // 상품수정
    @GetMapping("/products/{id}/edit")
    public String editProductForm(@PathVariable Long id,
                                  Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);
        Product product = productService.findWithCategoryTreeById(id);
        if (product.getUser() == null || !product.getUser().getId().equals(loginUser.getId())) {
            return "redirect:/access-denied";
        }
        model.addAttribute("productForm", ProductForm.from(product));
        // 속성/속성값 목록 추가 (DTO 변환)
        model.addAttribute("attributes", attributeRepository.findAll());
        List<AttributeValue> attributeValues = attributeValueRepository.findAllWithAttribute();
        List<AttributeValueDto> attributeValueDtos = attributeValues.stream()
                .map(av -> new AttributeValueDto(av.getId(), av.getValue(), av.getAttribute().getName()))
                .toList();
        model.addAttribute("attributeValues", attributeValueDtos);
        return "product/update";
    }

//    @PostMapping(value = "/products/{id}/edit", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
//    public String updateProduct(@PathVariable Long id,
//                                @RequestParam("productJson") String productJson,
//                                @RequestParam(value = "images", required = false) MultipartFile[] images,
//                                @RequestParam(value = "deleteIndexes", required = false) String deleteIndexes,
//                                Authentication authentication) {
//        String email = UserUtils.getEmail(authentication);
//        if (email == null) return "redirect:/user/login";
//        Users loginUser = (Users) UserUtils.getUser(authentication);
//        try {
//            // JSON -> ProductForm 변환
//            com.fasterxml.jackson.databind.ObjectMapper objectMapper = new com.fasterxml.jackson.databind.ObjectMapper();
//            ProductForm form = objectMapper.readValue(productJson, ProductForm.class);
//            // 서비스 호출
//            productService.updateProduct(id, form, images, deleteIndexes, loginUser);
//        } catch (Exception e) {
//            e.printStackTrace();
//            return "redirect:/error";
//        }
//        return "redirect:/products/" + id;
//    }

    @PostMapping(value = "/products/{id}/edit", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public String updateProduct(@PathVariable Long id,
                                @RequestParam("productJson") String productJson,
                                @RequestParam(value = "images", required = false) MultipartFile[] images,
                                Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);
        try {
            // JSON -> ProductForm 변환
            com.fasterxml.jackson.databind.ObjectMapper objectMapper = new com.fasterxml.jackson.databind.ObjectMapper();
            ProductForm form = objectMapper.readValue(productJson, ProductForm.class);

            // ✅ deleteIndexes는 이제 ProductForm 안에 포함되어 있음
            List<Integer> deleteIndexes = form.getDeleteIndexes();

            // 서비스 호출
            productService.updateProduct(id, form, images, deleteIndexes, loginUser);
        } catch (Exception e) {
            e.printStackTrace();
            return "redirect:/error";
        }
        return "redirect:/products/" + id;
    }


    /*
    // 방법2: @RequestPart 방식 (예비용)
    @PostMapping(value = "/products/{id}/edit", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public String updateProductAlt(@PathVariable Long id,
                                @RequestPart("productJson") ProductForm form,
                                @RequestPart(value = "images", required = false) List<MultipartFile> images,
                                @RequestParam(value = "deleteIndexes", required = false) String deleteIndexes,
                                Authentication authentication) {
        // ...
        return "redirect:/products/" + id;
    }
    */

    @PostMapping("/products/{id}/delete")
    public String deleteProduct(@PathVariable Long id,
                                Authentication authentication) throws Exception {
        String email = UserUtils.getEmail(authentication);
        Users loginUser = email != null ? (Users) UserUtils.getUser(authentication) : null;
        if (loginUser == null) {
            return "redirect:/user/user/login";
        }
        productService.deleteProduct(id, loginUser);
        return "redirect:/products";
    }
}