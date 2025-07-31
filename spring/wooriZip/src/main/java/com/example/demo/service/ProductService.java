package com.example.demo.service;

import com.example.demo.dto.ProductDetailDto;
import com.example.demo.dto.ProductForm;
import com.example.demo.dto.ProductListDto;
import com.example.demo.dto.ProductModelDto;

import com.example.demo.entity.*;
import com.example.demo.repository.CategoryRepository;
import com.example.demo.repository.ProductImageRepository;
import com.example.demo.repository.ProductRepository;
import com.example.demo.repository.WishlistRepository;
import com.example.demo.repository.AttributeValueRepository;
import com.example.demo.repository.ProductModelAttributeRepository;
import com.example.demo.repository.ProductModelRepository;
import com.example.demo.repository.OrderItemRepository;
import com.example.demo.repository.CartItemRepository;
import com.example.demo.repository.QnaPostRepository;
import com.example.demo.repository.ProductDetailRepository;
import com.example.demo.repository.ReviewPostRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.PageImpl;

@Slf4j
@RequiredArgsConstructor
@Service
public class ProductService {

    private final ProductRepository productRepository;
    private final ProductImageRepository imageRepository;
    private final WishlistRepository wishlistRepository;
    private final CategoryRepository categoryRepository;
    private final AttributeValueRepository attributeValueRepository;
    private final ProductModelAttributeRepository productModelAttributeRepository;
    private final ProductModelRepository productModelRepository;
    private final OrderItemRepository orderItemRepository;
    private final CartItemRepository cartItemRepository;
    private final QnaPostRepository qnaPostRepository;
    private final ProductDetailRepository productDetailRepository;
    private final ReviewPostRepository reviewPostRepository;

    // ✅ 상품 등록
    public Long createProduct(ProductForm form,
                              List<MultipartFile> productImgFileList,  // List<MultipartFile>로 수정
                              Users loginUser) throws Exception {
        log.info("상품 등록 시작: {}", form);

        // 1. 카테고리 조회
        Category category = categoryRepository.findById(form.getCategoryId())
                .orElseThrow(() -> new IllegalArgumentException("카테고리를 찾을 수 없습니다."));

        // 2. Product 생성
        Product product = form.createProduct(category, loginUser);
        // product = productRepository.save(product);  // 상품 저장 (중복 저장 제거)

        // 3. 모델 생성 및 재고 계산
        int totalStock = 0;
        List<ProductModelDto> validModels = form.getProductModelDtoList().stream()
                .filter(dto -> dto.getProductModelSelect() != null && dto.getPrice() != null && dto.getPrStock() != null)
                .collect(Collectors.toList());
        List<ProductModel> modelList = new ArrayList<>();
        for (ProductModelDto dto : validModels) {
            ProductModel model = new ProductModel();
            model.setProductModelSelect(dto.getProductModelSelect());  // 옵션명(자유입력)
            model.setPrice(dto.getPrice());  // 가격 설정
            model.setPrStock(dto.getPrStock());  // 재고 설정
            product.addProductModel(model);  // 상품에 모델 추가
            modelList.add(model);
            totalStock += model.getPrStock();
        }

        product.setStockQuantity(totalStock); // 총 재고량 업데이트

        // 옵션(모델) 중 가장 저렴한 가격을 상품 대표 가격으로 세팅
        if (!product.getProductModels().isEmpty()) {
            int minPrice = product.getProductModels().stream()
                    .mapToInt(ProductModel::getPrice)
                    .min()
                    .orElse(0);
            product.setPrice(minPrice);
        }

        // 1차 저장: Product와 ProductModel(옵션) 먼저 저장 (id 할당)
        productRepository.save(product);

        // 2차 저장: 옵션별 속성값(ProductModelAttribute) 연결 및 DB 저장
        for (int i = 0; i < validModels.size(); i++) {
            ProductModelDto dto = validModels.get(i);
            ProductModel savedModel = product.getProductModels().get(i); // id가 할당된 영속 모델
            if (dto.getAttributeValueIds() != null && !dto.getAttributeValueIds().isEmpty()) {
                for (Long attrValueId : dto.getAttributeValueIds()) {
                    AttributeValue attrValue = attributeValueRepository.findById(attrValueId)
                            .orElseThrow(() -> new IllegalArgumentException("옵션 속성값이 존재하지 않습니다."));
                    ProductModelAttribute pma = new ProductModelAttribute();
                    pma.setProductModel(savedModel); // 반드시 영속 모델로 연결
                    pma.setAttributeValue(attrValue);
                    savedModel.getModelAttributes().add(pma); // 옵션에 속성값 연결
                    productModelAttributeRepository.save(pma); // DB에 저장
                }
            }
        }

        // 4. 이미지 업로드 처리
        if (productImgFileList != null && !productImgFileList.isEmpty()) {
            List<String> imagePaths = handleAndReturnFiles(productImgFileList.toArray(new MultipartFile[0]));
            for (String path : imagePaths) {
                ProductImage image = new ProductImage();
                image.setImageUrl(path);
                image.setProduct(product);
                product.getImages().add(image); // product에 이미지 추가
                // imageRepository.save(image); // 별도 저장 불필요 (cascade)
            }
        }

        product = productRepository.save(product);  // 연관 엔티티까지 한 번만 저장

        return product.getId();  // 상품 ID만 반환
    }

    // 이미지 저장 유틸 (변경 없음)
    private List<String> handleAndReturnFiles(MultipartFile[] files) {
        List<String> filePaths = new ArrayList<>();
        String uploadDir = System.getProperty("user.dir") + "/uploads/";

        File dir = new File(uploadDir);
        if (!dir.exists()) dir.mkdirs();

        for (MultipartFile file : files) {
            if (!file.isEmpty()) {
                try {
                    String originalName = file.getOriginalFilename();
                    String uniqueName = System.currentTimeMillis() + "_" + originalName;
                    File dest = new File(uploadDir + uniqueName);
                    file.transferTo(dest);

                    filePaths.add("/uploads/" + uniqueName);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return filePaths;
    }

    // ✅ 상품 목록 조회
//    public List<Product> findProducts(Long categoryId) {
//        List<Product> products;
//        if (categoryId == null) {
//            products = productRepository.findAll(); // 아무 카테고리도 선택 안 했을 경우
//        } else {
//        // 선택한 카테고리와 그 하위 카테고리 포함하여 검색
//        List<Long> idsToSearch = new ArrayList<>();
//        idsToSearch.add(categoryId);
//        idsToSearch.addAll(getChildCategoryIds(categoryId));
//            products = productRepository.findByCategoryIdIn(idsToSearch);
//        }
//
//        // 각 상품의 평균 평점 계산 및 업데이트
//        for (Product product : products) {
//            double avgRating = reviewPostRepository.findByProductId(product.getId())
//                    .stream()
//                    .mapToInt(ReviewPost::getRating)
//                    .average()
//                    .orElse(0.0);
//            product.setAverageRating(avgRating);
//        }
//
//        return products;
//    }

    // ✅ 상품 목록 조회
    public List<ProductListDto> findProducts(Long categoryId) {
        List<Product> products;
        if (categoryId == null) {
            products = productRepository.findAll();
        } else {
            List<Long> idsToSearch = new ArrayList<>();
            idsToSearch.add(categoryId);
            idsToSearch.addAll(getChildCategoryIds(categoryId));
            products = productRepository.findByCategoryIdIn(idsToSearch);
        }

        // 평균 평점, 리뷰 수 포함하여 DTO 변환
        return products.stream().map(product -> {
            double avgRating = reviewPostRepository.findByProductId(product.getId())
                    .stream()
                    .mapToInt(ReviewPost::getRating)
                    .average()
                    .orElse(0.0);

            int reviewCount = reviewPostRepository.countByProductId(product.getId());

            return new ProductListDto(product, avgRating, reviewCount);
        }).toList();
    }

    // ✅ 페이징을 지원하는 상품 목록 조회
    public Page<ProductListDto> findProductsWithPaging(Long categoryId, int page, int size) {
        Pageable pageable = PageRequest.of(page - 1, size);
        Page<Product> productPage;
        
        if (categoryId == null) {
            // ✅ 삭제되지 않은 상품만
            productPage = productRepository.findByIsDeletedFalse(pageable);
        } else {
            List<Long> idsToSearch = new ArrayList<>();
            idsToSearch.add(categoryId);
            idsToSearch.addAll(getChildCategoryIds(categoryId));
            // ✅ 삭제되지 않은 상품만
            List<Product> allProducts = productRepository.findByCategoryIdInAndIsDeletedFalse(idsToSearch);
            int start = (page - 1) * size;
            int end = Math.min(start + size, allProducts.size());
            List<Product> pagedProducts = allProducts.subList(start, end);
            productPage = new PageImpl<>(pagedProducts, pageable, allProducts.size());
        }

        // 평균 평점, 리뷰 수 포함하여 DTO 변환
        List<ProductListDto> dtoList = productPage.getContent().stream().map(product -> {
            double avgRating = reviewPostRepository.findByProductId(product.getId())
                    .stream()
                    .mapToInt(ReviewPost::getRating)
                    .average()
                    .orElse(0.0);

            int reviewCount = reviewPostRepository.countByProductId(product.getId());

            return new ProductListDto(product, avgRating, reviewCount);
        }).toList();

        return new PageImpl<>(dtoList, pageable, productPage.getTotalElements());
    }

    public List<Product> findAllProducts() {
        return productRepository.findAll();
    }

    // 선택된 카테고리의 모든 하위 카테고리 ID를 재귀적으로 가져옴
    private List<Long> getChildCategoryIds(Long parentId) {
        List<Category> children = categoryRepository.findByParentId(parentId);
        List<Long> result = new ArrayList<>();

        for (Category child : children) {
            result.add(child.getId());
            result.addAll(getChildCategoryIds(child.getId())); // 재귀 호출
        }

        return result;
    }

    // ✅ 상품 상세 조회 (찜 여부 포함)
    public ProductDetailDto getProductDetail(Long productId, Users user) {
        // 1. 상품 조회
        Product product = productRepository.findById(productId)
                .orElseThrow(() -> new IllegalArgumentException("상품이 존재하지 않습니다."));

        // 2. 평균 평점 계산 및 업데이트
        double avgRating = reviewPostRepository.findByProductId(productId)
                .stream()
                .mapToInt(ReviewPost::getRating)
                .average()
                .orElse(0.0);
        product.setAverageRating(avgRating);

        // 3. 사용자가 찜한 상품 여부 체크
        boolean liked = user != null && wishlistRepository.existsByUserAndProduct(user, product);

        // 3. productId로 모든 ProductModel 조회
        List<ProductModel> allModels = productModelRepository.findAllByProduct_ProductId(productId);

        List<ProductModelDto> modelDtos = allModels.stream()
                .map(model -> {
                    ProductModelDto dto = new ProductModelDto();
                    dto.setId(model.getId()); // ID 매핑 확인
                    dto.setProductModelSelect(model.getProductModelSelect());// 모델명(자유입력)
                    dto.setPrice(model.getPrice()); // 가격
                    dto.setPrStock(model.getPrStock()); // 재고
                    if (model.getModelAttributes() != null && !model.getModelAttributes().isEmpty()) {
                        java.util.List<Long> attrValueIds = model.getModelAttributes().stream()
                                .map(pma -> pma.getAttributeValue().getId())
                                .collect(Collectors.toList());
                        dto.setAttributeValueIds(attrValueIds);
                        try {
                            ObjectMapper objectMapper = new ObjectMapper();
                            dto.setAttributeValueIdsStr(objectMapper.writeValueAsString(attrValueIds));
                        } catch (JsonProcessingException e) {
                            log.error("Error converting attrValueIds to JSON string", e);
                            dto.setAttributeValueIdsStr(""); // 변환 실패 시 빈 문자열
                        }
                    } else {
                        dto.setAttributeValueIds(new java.util.ArrayList<>());
                        dto.setAttributeValueIdsStr("[]"); // 빈 배열이면 "[]"로 설정
                    }
                    return dto;
                }).collect(Collectors.toList());

        // 4. ProductDetailDto 생성 후 모델 정보 설정
        ProductDetailDto detailDto = new ProductDetailDto(product, liked);
        detailDto.setProductModels(modelDtos);  // 모델 정보 추가

        // 5. 반환
        return detailDto;
    }

    // ✅ 상품 수정
    @Transactional
    public void updateProduct(Long productId, ProductForm form, MultipartFile[] images, List<Integer> deleteIndexes, Users loginUser) {
        Product product = productRepository.findById(productId).orElseThrow();

        // 권한 체크
        if (!product.getUser().getId().equals(loginUser.getId())) {
            throw new AccessDeniedException("권한 없음");
        }

        // ✅ 기존 이미지 삭제
        if (deleteIndexes != null && !deleteIndexes.isEmpty()) {
            List<ProductImage> currentImages = product.getImages();
            deleteIndexes.stream()
                    .sorted(Comparator.reverseOrder())
                    .forEach(idx -> {
                        if (idx >= 0 && idx < currentImages.size()) {
                            ProductImage removed = currentImages.remove((int) idx);
                            deleteFile(removed.getImageUrl());
                            imageRepository.delete(removed);
                        }
                    });
        }

        // ✅ 새 이미지 업로드
        if (images != null) {
            List<String> imagePaths = handleAndReturnFiles(images);
            for (String path : imagePaths) {
                ProductImage image = new ProductImage();
                image.setImageUrl(path);
                image.setProduct(product);
                product.getImages().add(image);
            }
        }

        product.setName(form.getName());
        product.setPrice(form.getPrice());

        if (form.getCategoryId() != null) {
            Category category = categoryRepository.findById(form.getCategoryId())
                    .orElseThrow(() -> new IllegalArgumentException("유효하지 않은 카테고리 ID"));
            product.setCategory(category);
        }

        // ✅ 기존 옵션 Map으로 구성
        Map<Long, ProductModel> existingModelMap = product.getProductModels().stream()
                .collect(Collectors.toMap(ProductModel::getId, pm -> pm));

        if (form.getProductModelDtoList() != null && !form.getProductModelDtoList().isEmpty()) {

            // ✅ 삭제 대상 추출을 위한 ID Set 생성
            Set<Long> incomingModelIds = form.getProductModelDtoList().stream()
                    .map(ProductModelDto::getId)
                    .filter(Objects::nonNull)
                    .collect(Collectors.toSet());

            // ✅ 삭제 대상 추출
            for (ProductModel existingModel : new ArrayList<>(product.getProductModels())) {
                Long id = existingModel.getId();
                if (!incomingModelIds.contains(id)) {
                    if (orderItemRepository.existsByProductModelId(id)) {
                        existingModel.setDeleted(true);  // soft delete
                    } else {
                        product.getProductModels().remove(existingModel);  // 관계 제거
                        productModelRepository.delete(existingModel);      // 실제 삭제
                    }
                }
            }

            // ✅ 수량 초기화 후 재계산
            product.setStockQuantity(0);

            // ✅ 업데이트 또는 새로 추가
            for (ProductModelDto dto : form.getProductModelDtoList()) {
                if (dto.getId() != null) {
                    ProductModel existing = existingModelMap.get(dto.getId());
                    if (existing != null) {
                        existing.setProductModelSelect(dto.getProductModelSelect());
                        existing.setPrice(dto.getPrice());
                        existing.setPrStock(dto.getPrStock());
                        existing.setDeleted(false); // soft-delete 복구 가능성
                        product.setStockQuantity(product.getStockQuantity() + dto.getPrStock());
                    }
                } else {
                    ProductModel newModel = new ProductModel();
                    newModel.setProductModelSelect(dto.getProductModelSelect());
                    newModel.setPrice(dto.getPrice());
                    newModel.setPrStock(dto.getPrStock());
                    newModel.setProduct(product);
                    product.addProductModel(newModel);
                    product.setStockQuantity(product.getStockQuantity() + dto.getPrStock());
                }
            }
        }
        productRepository.save(product);
    }

    // ✅ 파일 삭제 유틸
    private void deleteFile(String relativePath) {
        if (relativePath == null) return;
        String absolutePath = System.getProperty("user.dir") + relativePath;
        File file = new File(absolutePath);
        if (file.exists()) file.delete();
    }

    public Product findById(Long id) {
        return productRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("상품이 존재하지 않습니다."));
    }

    // 상품 + 카테고리 계층 fetch join
    public Product findWithCategoryTreeById(Long id) {
        return productRepository.findWithCategoryTreeById(id);
    }

//    @Transactional
//    public void deleteProduct(Long id, Users user) {
//        Product product = productRepository.findById(id)
//                .orElseThrow(() -> new IllegalArgumentException("상품이 존재하지 않습니다."));
//
//        if (product.getUser() == null || !product.getUser().getId().equals(user.getId())) {
//            throw new AccessDeniedException("상품 삭제 권한이 없습니다.");
//        }
//
//        // 1. 옵션(모델) 목록
//        List<ProductModel> productModels = product.getProductModels();
//
//        // 2. order_item에서 해당 옵션을 참조하는 row 삭제
//        orderItemRepository.deleteByProductModelIn(productModels);
//
//        // 3. wishlist에서 해당 상품을 참조하는 row 삭제
//        wishlistRepository.deleteByProduct(product);
//
//        // 4. cart_item에서 해당 상품을 참조하는 row 삭제
//        cartItemRepository.deleteByProduct(product);
//
//        // 5. qna_post에서 해당 상품을 참조하는 row 삭제
//        qnaPostRepository.deleteByProduct(product);
//
//        // 6. product_detail에서 해당 상품을 참조하는 row 삭제
//        productDetailRepository.deleteByProduct(product);
//
//        // 7. review_post에서 해당 상품을 참조하는 row 삭제
//        reviewPostRepository.deleteByProduct(product);
//
//        // 8. 이미지, 옵션 등 연관 엔티티 삭제 (기존 코드)
//        imageRepository.deleteAll(product.getImages());
//        product.getProductModels().clear();
//
//        // 9. 상품 삭제
//        productRepository.delete(product);
//
//    }

    //soft delete로 바꾼 로직
    @Transactional
    public void deleteProduct(Long id, Users user) {
        Product product = productRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("상품이 존재하지 않습니다."));

        if (product.getUser() == null || !product.getUser().getId().equals(user.getId())) {
            throw new AccessDeniedException("상품 삭제 권한이 없습니다.");
        }

        // 👉 옵션들도 soft delete
        for (ProductModel model : product.getProductModels()) {
            model.setDeleted(true);
        }

        // 👉 상품도 soft delete
        product.setDeleted(true);
    }

    public List<Product> findByUser(Long userId) {
        List<Long> myWishList = wishlistRepository.findProductByUser(userId);
        if(myWishList.isEmpty()) {
            return Collections.emptyList();
        }
        return productRepository.findByIdIn(myWishList);
    }

    public List<Product> findAll() {
        return productRepository.findAll();
    }
}
