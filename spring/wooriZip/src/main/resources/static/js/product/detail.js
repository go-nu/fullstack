document.addEventListener("DOMContentLoaded", function () {
    // 가격 계산
    function calculateTotalPrice() {
        var count = parseInt($('#count').val());
        var price = parseInt($('#price').val());
        if (!isNaN(count) && !isNaN(price)) {
            var totalPrice = count * price;
            $('#totalPrice').html(totalPrice + '원');
        } else {
            $('#totalPrice').html('모델/옵션을 선택하세요.');
        }
    }

    calculateTotalPrice();
    $('#count').change(calculateTotalPrice);

    // 모델 선택 이벤트
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', function () {
            const selected = this.selectedOptions[0];
            if (selected && selected.value) {
                document.getElementById('modelInfo').innerText =
                    `가격: ${selected.dataset.price}원, 재고: ${selected.dataset.stock}`;
            } else {
                document.getElementById('modelInfo').innerText = '';
            }
        });
    }

    // submitForm 정의
    window.submitForm = function (action) {
        const form = document.getElementById('productForm');

        // 기존 items 필드 제거
        ['items[0].modelId', 'items[0].count'].forEach(name => {
            const old = form.querySelector(`input[name="${name}"]`);
            if (old) old.remove();
        });

        // ✅ select 박스에서 modelId 가져오기
        const selectedModelId = document.getElementById('modelSelect')?.value;
        const count = document.getElementById('count').value;
        
        if (!selectedModelId || !count) {
            alert("옵션과 수량을 선택해주세요.");
            return;
        }
        
        // 동적 필드 추가
        const modelInput = document.createElement("input");
        modelInput.type = "hidden";
        modelInput.name = "items[0].modelId";
        modelInput.value = selectedModelId;
        form.appendChild(modelInput);

        const countInput = document.createElement("input");
        countInput.type = "hidden";
        countInput.name = "items[0].count";
        countInput.value = count;
        form.appendChild(countInput);

        // 액션 설정
        if (action === 'cart') {
            form.action = "/cart/add";
            document.getElementById('actionType').value = 'cart';
        } else if (action === 'buy') {
            form.action = "/order/now";
            document.getElementById('actionType').value = 'buy';
        }
        
        form.submit();
    };

    // 탭 전환 - 모든 탭 전환 로직을 여기로 통합
    function initializeTabs() {
        // 탭 전환 함수
        function switchTab(targetId) {
            document.querySelectorAll('.tab-button').forEach(b => {
                b.classList.remove('active');
                if (b.getAttribute('data-target') === targetId) {
                    b.classList.add('active');
                }
            });

            document.querySelectorAll('.tab-content-section').forEach(section => {
                section.style.display = section.id === targetId ? 'block' : 'none';
            });
        }

        // 탭 버튼 클릭 이벤트
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.addEventListener('click', function () {
                const target = this.getAttribute('data-target');
                switchTab(target);
                // URL 해시 업데이트 (페이지 정보 유지)
                const currentHash = window.location.hash;
                const pageMatch = currentHash.match(/page-(\d+)/);
                const pageInfo = pageMatch ? `,page-${pageMatch[1]}` : '';
                window.location.hash = target === 'qna-section' ? `qna-tab${pageInfo}` : 'review-section';
            });
        });

        // URL 해시에 따른 탭 전환 및 페이지 처리
        function handleHash() {
            const hash = window.location.hash;
            if (hash.includes('qna-tab')) {
                switchTab('qna-section');
                
                // 페이지 번호 처리
                const pageMatch = hash.match(/page-(\d+)/);
                if (pageMatch) {
                    const pageNum = pageMatch[1];
                    const pageButton = document.querySelector(`.page-link[data-page="${pageNum}"]`);
                    if (pageButton) {
                        pageButton.click();
                    }
                }

                // QNA 항목으로 스크롤
                const qnaMatch = hash.match(/qna-(\d+)/);
                if (qnaMatch) {
                    setTimeout(() => {
                        const qnaElement = document.querySelector(`#qna-${qnaMatch[1]}`);
                        if (qnaElement) {
                            qnaElement.scrollIntoView({ behavior: 'smooth' });
                        }
                    }, 500); // 페이지 전환 후 스크롤하기 위해 약간의 지연 추가
                }
            }
        }

        // 초기 로드 및 해시 변경 시 처리
        window.addEventListener('hashchange', handleHash);
        handleHash();
    }

    // 탭 초기화 실행
    initializeTabs();
});
