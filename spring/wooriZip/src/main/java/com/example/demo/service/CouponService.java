package com.example.demo.service;

import com.example.demo.dto.CouponDto;
import com.example.demo.entity.Coupon;
import com.example.demo.entity.UserCoupon;
import com.example.demo.entity.Users;
import com.example.demo.repository.CouponRepository;
import com.example.demo.repository.UserCouponRepository;
import com.example.demo.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class CouponService {

    private final UserRepository userRepository;
    private final CouponRepository couponRepository;
    private final UserCouponRepository userCouponRepository;

    public void create(CouponDto dto) {
        Coupon coupon = Coupon.createCoupon(dto);
        couponRepository.save(coupon);
    }

    public List<CouponDto> getAllCoupons() {
        return couponRepository.findAll().stream()
                .map(coupon -> CouponDto.builder()
                        .id(coupon.getId())
                        .name(coupon.getName())
                        .code(coupon.getCode())
                        .type(coupon.getType())
                        .discountAmount(coupon.getDiscountAmount())
                        .discountPercent(coupon.getDiscountPercent())
                        .startDate(coupon.getStartDate())
                        .endDate(coupon.getEndDate())
                        .isActive(coupon.isActive())
                        .minOrderPrice(coupon.getMinOrderPrice())
                        .usageLimit(coupon.getUsageLimit())
                        .build())
                        .collect(Collectors.toList());

    }

    public void claimCoupon(Long couponId, String email) {
        Coupon coupon = couponRepository.findById(couponId)
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 쿠폰입니다."));

        if (coupon.getUsageLimit() <= 0) {
            throw new IllegalStateException("쿠폰 수량이 모두 소진되었습니다.");
        }

        Users user = userRepository.findByEmail(email)
                .orElseThrow(() -> new IllegalArgumentException("유저 정보를 찾을 수 없습니다."));

        boolean alreadyClaimed = userCouponRepository.existsByUserAndCoupon(user, coupon);
        if (alreadyClaimed) {
            throw new IllegalStateException("이미 해당 쿠폰을 받았습니다.");
        }

        // 수량 감소
        coupon.setUsageLimit(coupon.getUsageLimit() - 1);
        couponRepository.save(coupon);

        // 사용자에게 쿠폰 발급
        UserCoupon userCoupon = UserCoupon.builder()
                .user(user)
                .coupon(coupon)
                .used(false)
                .build();

        userCouponRepository.save(userCoupon);
    }


}
