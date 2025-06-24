package com.shop.repository;

import com.shop.entity.Order;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface OrderRepository extends JpaRepository<Order, Long> {

    // order 엔티티를 o로 조회
    // 주어진 파라미터와 일치하는 member 엔티티의 email을 기준으로 연관된 order 조회
    // 날짜 내림차순 정렬
    @Query("select o from Order o where o.member.email = :email order by o.orderDate desc")
    List<Order> findOrders(@Param("email") String email, Pageable pageable);

    /*@Query(value = "SELECT o.* FROM orders o " +
        "JOIN members m ON o.member_id = m.id " +
        "WHERE m.email = :email " +
        "ORDER BY o.order_date DESC",
        nativeQuery = true)
    List<Order> findOrders(@Param("email") String email, Pageable pageable);*/

    @Query("select count(o) from Order o where o.member.email = :email")
    Long countOrder(@Param("email") String email);
}
