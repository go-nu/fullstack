package com.example.demo.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class UserCoupon {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    private Users user;

    @ManyToOne
    private Coupon coupon;

    @Column(nullable = false)
    private boolean used;

    private LocalDateTime assignedAt;

    @Builder
    public UserCoupon(Users user, Coupon coupon, boolean used) {
        this.user = user;
        this.coupon = coupon;
        this.used = used;
        this.assignedAt = LocalDateTime.now();
    }
}
