function generateRandomCode() {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let code = '';
    for (let i = 0; i < 10; i++) {
        code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    document.getElementById("code").value = code;
}

function toggleDiscountInputs() {
    const amountInput = document.getElementById("amountInput");
    const percentInput = document.getElementById("percentInput");

    const typeAmount = document.getElementById("typeAmount").checked;
    const typePercent = document.getElementById("typePercent").checked;

    amountInput.style.display = typeAmount ? "block" : "none";
    percentInput.style.display = typePercent ? "block" : "none";
}

window.onload = toggleDiscountInputs;

// ✅ 쿠폰 활성화/비활성화 상태 토글 함수 (boolean 사용)
function toggleCouponStatus(checkbox) {
    const couponId = checkbox.getAttribute('data-id');
    const isActive = checkbox.checked;

    fetch(`/admin/coupons/${couponId}/status`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ isActive: isActive })
    })
    .then(response => {
        if (!response.ok) throw new Error('상태 변경 실패');
        return response.json();
    })
    .then(data => {
        // ✅ 상태 텍스트를 즉시 변경
        const labelSpan = checkbox.closest('td').querySelector('.form-check-label span');
        if (labelSpan) {
            labelSpan.textContent = isActive ? '활성화' : '비활성화';
        }
    })
    .catch(error => {
        alert('오류 발생: ' + error.message);
        checkbox.checked = !checkbox.checked; // 실패 시 원상복구
    });
}


