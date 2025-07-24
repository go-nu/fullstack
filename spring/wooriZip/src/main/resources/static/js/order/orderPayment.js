$(document).ready(function () {
console.log($);
    // 상품 수량 변동에 따른 처리
    $('.item-count').on('change', function () {
        let $input = $(this);
        let newCount = $input.val();
        let orderDetailId = $input.data('orderDto-detail-id');

        $.ajax({
            url: '/order/updateQuantity',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({orderDetailId: orderDetailId, count: newCount}),
            success: function (response) {
                if (response.success) {
                    console.log('수량 업데이트 성공');
                    location.reload(); // 성공 시 페이지 새로고침
                } else {
                    console.error('수량 업데이트 실패');
                    alert('수량 업데이트 실패');
                }
            },
            error: function (xhr, status, error) {
                console.error('에러 발생:', error);
                alert('에러 발생');
                location.reload();
            }
        });
    });

    // 주문 요청
    // 결제하기 버튼 클릭 시
    $('#submit-orderDto').on('click', function (e) {
        e.preventDefault();

        $.ajax({
            url: '/pay/checkout',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({orderId: $('input[name="orderId"]').val()}),
            success: function (response) {
                if (response.success) {
                    window.location.href = response.redirectUrl; // 토스 결제 페이지로 리다이렉트
                } else {
                    alert('오류: ' + response.message);
                }
            },
            error: function (xhr) {
                let response = JSON.parse(xhr.responseText);
                alert('오류: ' + response.message);
            }
        });
    });

        // 페이지 떠날 때 이벤트 처리
        window.addEventListener('beforeunload', function (e) {
            if (!isPaymentSuccess) { // 결제 한것이 아니면 cancel 로 이동
                fetch('/cancel', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({orderId: order.orderID})
                }).then(response => {
                    if (!response.ok) {
                        console.error('Failed to cancel order');
                    }
                }).catch(error => {
                    console.error('Error canceling order:', error);
                });

                // 사용자에게 경고 메시지 표시
                e.preventDefault();
                e.returnValue = '주문을 취소합니다.';
            }
        });
});

// 전화번호 합치기
window.addEventListener("DOMContentLoaded", function () {
    let fullPhone = /*[[${orderDto.phone}]]*/ '';

    // null, undefined, 'null' 문자열 등 예외 처리
    if (fullPhone == null || fullPhone === 'null' || fullPhone.trim() === '') {
        fullPhone = '';
    } else {
        fullPhone = fullPhone.replace(/[^0-9]/g, ''); // 숫자만 추출
    }

    if (fullPhone && fullPhone.length === 11) {
        document.getElementById("phone1").value = fullPhone.substring(0, 3);
        document.getElementById("phone2").value = fullPhone.substring(3, 7);
        document.getElementById("phone3").value = fullPhone.substring(7);
    }

    function updatePhoneValue() {
        const phone1 = document.getElementById("phone1").value;
        const phone2 = document.getElementById("phone2").value;
        const phone3 = document.getElementById("phone3").value;
        document.getElementById("phone").value = phone1 + phone2 + phone3;
    }
    function validatePhoneNumber() {
        updatePhoneValue();
        const phone = document.getElementById('phone').value;
        if (phone.length !== 11) {
            alert('전화번호를 다시 한 번 확인해 주세요.');
            document.getElementById('phone1').focus();
            return false;
        }
        return true;
    }

    document.getElementById("phone1").addEventListener("input", updatePhoneValue);
    document.getElementById("phone2").addEventListener("input", updatePhoneValue);
    document.getElementById("phone3").addEventListener("input", updatePhoneValue);
});

//이메일 입력
document.addEventListener("DOMContentLoaded", function () {
    const domainInput = document.getElementById("email-domain");
    const domainSelect = document.getElementById("domain-select");

    // select 선택 시 → input 값을 해당 도메인으로 변경
    domainSelect.addEventListener("change", function () {
        if (this.value !== "") {
            domainInput.value = this.value;
        } else {
            domainInput.value = "";
        }
    });

    // input 수동 입력 시 select 값을 "직접입력"으로 되돌림
    domainInput.addEventListener("input", function () {
    domainSelect.value = "";
    });
});