$(document).ready(function () {

    // ✅ 수량 변경 시 서버에 반영하고 요약만 갱신
    $('.item-count').on('change', function () {
        let $input = $(this);
        let newCount = $input.val();
        let cartId = $input.data('cart-id');

        $.ajax({
            url: '/cart/update',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({cartItemId: cartId, count: newCount}),
            success: function (response) {
                if (response.success) {
                    updateSummary(); // 페이지 새로고침 없이 요약만 업데이트
                } else {
                    alert('오류: ' + response.message);
                }
            },
            error: function (xhr) {
                let response = JSON.parse(xhr.responseText);
                alert('오류: ' + response.message);
                location.reload();
            }
        });
    });

    // ✅ 전체 주문 클릭 시 전체 선택 후 전송
    $(".allOrder").on("click", function (event) {
        event.preventDefault();
        $(".cart-item-checkbox").prop("checked", true);
        sendOrderRequest();
    });

    // ✅ 선택 주문 클릭 시 선택된 항목만 전송
    $(".selectOrder").on("click", function (event) {
        event.preventDefault();
        sendOrderRequest();
    });

    // ✅ 전체 선택/해제 체크박스
    $("#selectAll").on("change", function () {
        $(".cart-item-checkbox").prop("checked", $(this).prop("checked"));
        updateSummary(); // 전체 선택/해제 시 요약 반영
    });

    // ✅ 수량 or 체크박스 변경 시 요약 갱신
    $(".cart-item-checkbox, .item-count").on("change", function () {
        updateSummary();
    });

    // ✅ 페이지 진입 시 최초 1회 결제 요약 계산
    updateSummary();

    // ✅ 선택된 항목 기반 주문 전송 함수
    function sendOrderRequest() {
        let selectedItems = $(".cart-item-checkbox:checked").map(function () {
            return $(this).val();
        }).get();

        if (selectedItems.length === 0) {
            alert("주문할 상품을 선택하세요.");
            return;
        }

        $.ajax({
            url: '/pay/ready',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(selectedItems),
            success: function (response) {
                if (response.redirectUrl) {
                    window.location.href = response.redirectUrl;
                } else {
                    alert("결제 준비 실패");
                }
            },
            error: function () {
                alert("서버 오류 발생");
            }
        });
    }

    // ✅ 체크된 상품 기준 결제 요약 계산 함수
    function updateSummary() {
        let totalPrice = 0;
        let deliveryFee = 3000;

        $("tr").has("input.cart-item-checkbox").each(function () {
            const priceText = $(this).find("td").eq(4).text().replace(/[^0-9]/g, "");
            const count = parseInt($(this).find("input[name='count']").val());

            const price = parseInt(priceText);
            if (!isNaN(price) && !isNaN(count)) {
                const subtotal = price * count;

                // ✅ 합계 셀 업데이트 (index 5)
                $(this).find("td").eq(6).text(subtotal.toLocaleString() + "원");

                // ✅ 체크된 항목만 총합 계산
                if ($(this).find(".cart-item-checkbox").prop("checked")) {
                    totalPrice += subtotal;
                }
            }
        });

        $("#finalTotalPrice").text(totalPrice.toLocaleString() + "원");
        $("#deliveryFee").text((totalPrice === 0 ? 0 : deliveryFee).toLocaleString() + "원");
        $("#finalPaymentAmount").text((totalPrice + (totalPrice === 0 ? 0 : deliveryFee)).toLocaleString() + "원");
    }

    // 선택 항목 삭제
    $('.deleteSelected').click(function () {
        const selectedItems = [];

        // 체크된 항목의 cartItemId 수집
        $('.cart-item-checkbox:checked').each(function () {
            selectedItems.push($(this).val());
        });

        if (selectedItems.length === 0) {
            alert('삭제할 항목을 선택해주세요.');
            return;
        }

        // POST 요청 전송 (fetch 또는 jQuery AJAX 가능)
        $.ajax({
            type: 'POST',
            url: '/cart/deleteSelected',
            contentType: 'application/json',
            data: JSON.stringify(selectedItems),
            success: function (response) {
                // 성공 후 페이지 새로고침 또는 삭제된 항목 제거
                location.reload();
            },
            error: function (xhr) {
                alert('선택 삭제에 실패했습니다.');
            }
        });
    });

    // 전체 항목 삭제
    $('.deleteAll').click(function () {
        if (!confirm('장바구니의 모든 상품을 삭제하시겠습니까?')) return;

        $.ajax({
            type: 'POST',
            url: '/cart/clear',
            success: function () {
                location.reload();
            },
            error: function () {
                alert('전체 삭제에 실패했습니다.');
            }
        });
    });
});
