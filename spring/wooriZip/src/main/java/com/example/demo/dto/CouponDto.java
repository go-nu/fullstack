package com.example.demo.dto;

import com.example.demo.constant.CouponType;
import lombok.*;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CouponDto {

    private Long id;
    private String name;
    private String code;
    private CouponType type;

    private Integer discountAmount;
    private Integer discountPercent;

    private LocalDateTime startDate;
    private LocalDateTime endDate;

    private boolean isActive;
    private int minOrderPrice;
    private int usageLimit;

}
