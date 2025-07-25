package com.example.demo.controller;

import com.example.demo.dto.ProductForm;
import com.example.demo.entity.AttributeValue;
import com.example.demo.entity.Product;
import com.example.demo.entity.Users;
import com.example.demo.dto.AttributeValueDto;
import com.example.demo.dto.ProductModelDto;
import com.example.demo.repository.AttributeRepository;
import com.example.demo.repository.AttributeValueRepository;
import com.example.demo.service.*;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;
import com.example.demo.dto.ProductDetailDto;
import com.example.demo.dto.ProductDetailInfoDto;
import com.example.demo.dto.ReviewPostDto;
import com.example.demo.entity.ReviewPost;
import com.example.demo.entity.QnaPost;
import com.example.demo.dto.QnaPostDto;
import org.springframework.data.domain.Page;
import org.springframework.http.MediaType;

@Controller
@RequiredArgsConstructor
public class ProductController {
    private final ProductService productService;
    private final AttributeRepository attributeRepository;
    private final AttributeValueRepository attributeValueRepository;
    private final ProductDetailService productDetailService;
    private final QnaPostService qnaPostService;
    private final ReviewPostService reviewPostService;
    private final WishlistService wishlistService;

    @GetMapping("/admin/products")
    public String adminProductList(Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        model.addAttribute("loginUser", UserUtils.getUser(authentication));

        List<Product> products = productService.findProducts(null); // null = 모든 상품 조회
        model.addAttribute("products", products);
        return "product/admin-list";
    }

    @GetMapping("/products/form")
    public String showProductForm(Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        model.addAttribute("loginUser", UserUtils.getUser(authentication));

        ProductForm productForm = new ProductForm();
        ProductModelDto defaultModel = new ProductModelDto();
        productForm.getProductModelDtoList().add(defaultModel);

        model.addAttribute("productForm", productForm);
        model.addAttribute("attributes", attributeRepository.findAll());
        List<AttributeValue> attributeValues = attributeValueRepository.findAllWithAttribute();
        List<AttributeValueDto> attributeValueDtos = attributeValues.stream()
                .map(av -> new AttributeValueDto(av.getId(), av.getValue(), av.getAttribute().getName()))
                .toList();
        model.addAttribute("attributeValues", attributeValueDtos);
        return "product/products";
    }

    @PostMapping("/admin/products")
    public String createProduct(@ModelAttribute ProductForm form,
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
                ObjectMapper objectMapper = new ObjectMapper();
                List<ProductModelDto> modelDtoList = objectMapper.readValue(
                        modelsJson, new TypeReference<List<ProductModelDto>>() {}
                );
                form.setProductModelDtoList(modelDtoList);
            }

            // 상품 등록 처리
            Long productId = productService.createProduct(form, Arrays.asList(images), loginUser);
            model.addAttribute("productForm", form);

        } catch (Exception e) {
            e.printStackTrace();
            return "redirect:/error";
        }
        return "redirect:/products";
    }

    @GetMapping("/products")
    public String showProductList(@RequestParam(name = "category", required = false) Long categoryId,
                                  Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        Users loginUser = email != null ? (Users) UserUtils.getUser(authentication) : null;
        model.addAttribute("loginUser", loginUser);
        List<Product> productList = productService.findProducts(categoryId);
        model.addAttribute("products", productList);
        return "product/list";
    }

    @GetMapping("/admin/products/register")
    public String showProductRegisterForm(Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        model.addAttribute("loginUser", UserUtils.getUser(authentication));

        ProductForm productForm = new ProductForm();
        ProductModelDto defaultModel = new ProductModelDto();
        productForm.getProductModelDtoList().add(defaultModel);

        model.addAttribute("productForm", productForm);
        model.addAttribute("attributes", attributeRepository.findAll());
        List<AttributeValue> attributeValues = attributeValueRepository.findAllWithAttribute();
        List<AttributeValueDto> attributeValueDtos = attributeValues.stream()
                .map(av -> new AttributeValueDto(av.getId(), av.getValue(), av.getAttribute().getName()))
                .toList();
        model.addAttribute("attributeValues", attributeValueDtos);
        return "product/products";
    }

    @PostMapping("/admin/products/register")
    public String registerProduct(@ModelAttribute ProductForm form,
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
                ObjectMapper objectMapper = new ObjectMapper();
                List<ProductModelDto> modelDtoList = objectMapper.readValue(
                        modelsJson, new TypeReference<List<ProductModelDto>>() {}
                );
                form.setProductModelDtoList(modelDtoList);
            }

            // 상품 등록 처리
            Long productId = productService.createProduct(form, Arrays.asList(images), loginUser);
            model.addAttribute("productForm", form);

        } catch (Exception e) {
            e.printStackTrace();
            return "redirect:/error";
        }
        return "redirect:/products";
    }

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
        ProductDetailInfoDto productDetail = productDetailService.findByProductId(id);
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
        // Product 엔티티의 averageRating 필드 사용
        Double averageRating = dto.getAverageRating();
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

    @GetMapping("/admin/products/{id}/update")
    public String showUpdateForm(@PathVariable Long id, Model model, Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        model.addAttribute("loginUser", UserUtils.getUser(authentication));

        Product product = productService.findById(id);
        if (product == null) {
            return "redirect:/admin/products";
        }

        ProductForm form = ProductForm.from(product);
        model.addAttribute("productForm", form);
        model.addAttribute("product", product);
        model.addAttribute("attributes", attributeRepository.findAll());
        List<AttributeValue> attributeValues = attributeValueRepository.findAllWithAttribute();
        List<AttributeValueDto> attributeValueDtos = attributeValues.stream()
                .map(av -> new AttributeValueDto(av.getId(), av.getValue(), av.getAttribute().getName()))
                .toList();
        model.addAttribute("attributeValues", attributeValueDtos);
        return "product/update";
    }

    @PostMapping(value = "/admin/products/{id}/update", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public String updateProduct(@PathVariable Long id,
                                @RequestParam("productJson") String productJson,
                                @RequestParam(value = "images", required = false) MultipartFile[] images,
                                Authentication authentication) {
        String email = UserUtils.getEmail(authentication);
        if (email == null) return "redirect:/user/login";
        Users loginUser = (Users) UserUtils.getUser(authentication);
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            ProductForm form = objectMapper.readValue(productJson, ProductForm.class);
            List<Integer> deleteIndexes = form.getDeleteIndexes();
            productService.updateProduct(id, form, images, deleteIndexes, loginUser);
        } catch (Exception e) {
            e.printStackTrace();
            return "redirect:/error";
        }
        return "redirect:/admin/products";
    }

    @PostMapping("/admin/products/delete/{id}")
    @ResponseBody
    public Map<String, Object> deleteProduct(@PathVariable Long id, Authentication authentication) {
        Map<String, Object> response = new HashMap<>();
        try {
            String email = UserUtils.getEmail(authentication);
            if (email == null) {
                response.put("success", false);
                response.put("message", "로그인이 필요합니다.");
                return response;
            }
            Users user = (Users) UserUtils.getUser(authentication);
            if (user == null) {
                response.put("success", false);
                response.put("message", "사용자를 찾을 수 없습니다.");
                return response;
            }

            productService.deleteProduct(id, user);
            response.put("success", true);
            response.put("message", "상품이 성공적으로 삭제되었습니다.");
        } catch (Exception e) {
            response.put("success", false);
            response.put("message", "상품 삭제 중 오류가 발생했습니다: " + e.getMessage());
        }
        return response;
    }
}