package com.example.demo.controller;

import com.example.demo.dto.*;
import com.example.demo.entity.*;
import com.example.demo.repository.CategoryRepository;
import com.example.demo.security.CustomUserDetails;
import com.example.demo.service.ProductService;
import com.example.demo.service.QnaPostService;
import com.example.demo.service.ReviewPostService;
import com.example.demo.service.WishlistService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;

@Controller
@RequiredArgsConstructor
@Slf4j
public class ProductController {

    private final ProductService productService;
    private final WishlistService wishlistService;
    private final ReviewPostService reviewPostService;
    private final QnaPostService qnaPostService;
    private final CategoryRepository categoryRepository;

    @GetMapping("/admin/products")
    public String showProductForm(Model model) {
        model.addAttribute("productForm", new ProductForm());
        return "product/products";
    }

    // 상품등록
    @PostMapping("/admin/products")
    public String createProduct(@ModelAttribute ProductForm form,
                                @RequestParam("images") MultipartFile[] images,
                                @AuthenticationPrincipal CustomUserDetails customUserDetails) {
        try {
            Users loginUser = customUserDetails.getUser();
            productService.createProduct(form, Arrays.asList(images), loginUser);
        } catch (Exception e) {
            e.printStackTrace();
            return "redirect:/error";
        }
        return "redirect:/products";
    }




    @GetMapping("/products")
    public String showProductList(@RequestParam(name = "category", required = false) Long categoryId, Model model) {
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
                              Model model,
                              @AuthenticationPrincipal CustomUserDetails customUserDetails) {

        Users user = customUserDetails != null ? customUserDetails.getUser() : null;

        // ───────────── 상품 상세 정보 ─────────────
        ProductDetailDto dto = productService.getProductDetail(id, user);
        model.addAttribute("product", dto);
        model.addAttribute("loginUser", user);

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

        model.addAttribute("reviewPage", reviewPage != null ? reviewPage : Page.empty());
        model.addAttribute("ratingSummary", ratingSummary != null ? ratingSummary : new HashMap<>());
        model.addAttribute("averageRating", averageRating != null ? averageRating : 0.0);
        model.addAttribute("hasWritten", hasWritten);
        model.addAttribute("sort", sort);

        // ───────────── QnA 데이터 ─────────────
        int qnaPageNum = Math.max(qnaPage - 1, 0);
        List<QnaPostDto> qnaList = qnaPostService.getQnaPostDtoList(id, qnaPageNum);
        long qnaTotal = qnaPostService.countByProduct(id);

        model.addAttribute("qnaList", qnaList != null ? qnaList : new ArrayList<>());
        model.addAttribute("qnaTotal", qnaTotal);
        model.addAttribute("qnaPage", qnaPage);

        return "product/detail";
    }




    @PostMapping("/wishlist/toggle")
    public String toggleWishlist(@RequestParam Long productId,
                                 @AuthenticationPrincipal CustomUserDetails customUserDetails) {
        Users user = customUserDetails != null ? customUserDetails.getUser() : null;
        if (user == null) {
            return "redirect:/user/login";
        }
        wishlistService.toggleWishlist(user, productId);
        return "redirect:/products/" + productId;
    }

    @GetMapping("/products/{id}/edit")
    public String editProductForm(@PathVariable Long id,
                                  @AuthenticationPrincipal CustomUserDetails customUserDetails,
                                  Model model) {
        Users loginUser = customUserDetails != null ? customUserDetails.getUser() : null;
        if (loginUser == null) {
            return "redirect:/user/login";
        }
        Product product = productService.findById(id);
        if (product.getUser() == null || !product.getUser().getId().equals(loginUser.getId())) {
            return "redirect:/access-denied";
        }
        model.addAttribute("productForm", ProductForm.from(product));
        return "product/update";
    }

    @PostMapping("/products/{id}/edit")
    public String updateProduct(@PathVariable Long id,
                                @ModelAttribute ProductForm form,
                                @RequestParam(value = "images", required = false) MultipartFile[] images,
                                @RequestParam(value = "deleteIndexes", required = false) String deleteIndexes,
                                @AuthenticationPrincipal CustomUserDetails customUserDetails) {
        Users loginUser = customUserDetails.getUser();
        productService.updateProduct(id, form, images, deleteIndexes, loginUser);
        return "redirect:/products/" + id;
    }

    @PostMapping("/products/{id}/delete")
    public String deleteProduct(@PathVariable Long id,
                                @AuthenticationPrincipal CustomUserDetails customUserDetails) throws Exception {
        Users loginUser = customUserDetails != null ? customUserDetails.getUser() : null;
        if (loginUser == null) {
            return "redirect:/user/login";
        }
        productService.deleteProduct(id, loginUser);
        return "redirect:/products";
    }
}
