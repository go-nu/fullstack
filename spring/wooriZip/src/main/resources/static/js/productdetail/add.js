let selectedFiles = [];
let removedExistingImages = [];

document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('previewContainer');
    const saveBtn = document.getElementById('saveBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const detailInfoTextarea = document.getElementById('detailInfo');

    // 기본 템플릿 설정
    const defaultTemplate = `품명: 
KC 인증정보: 
색상: 
구성품: 
주요 소재: 
제조자/수입자: 
제조국: 
크기: 
재공급 사유 및 하자 부위: 
배송/설치비용: 
품질보증기준: 
A/S 책임자와 전화번호: `;

    // 페이지 로드 시 상세정보 설정
    const existingDetailInfo = window.existingDetailInfo;
    if (existingDetailInfo && existingDetailInfo.trim() !== '') {
        // 기존 상세정보가 있으면 그것을 사용
        detailInfoTextarea.value = existingDetailInfo;
    } else {
        // 기존 상세정보가 없으면 기본 템플릿 사용
        detailInfoTextarea.value = defaultTemplate;
    }

    // 새 이미지 선택 시 처리 (인테리어 게시판 방식)
    imageInput.addEventListener('change', function() {
        const newFiles = Array.from(this.files);
        
        if (selectedFiles.length + newFiles.length > 20) {
            alert("이미지는 최대 20장까지 첨부할 수 있습니다.");
            return;
        }
        
        newFiles.forEach(file => {
            selectedFiles.push(file);
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const imageContainer = document.createElement('div');
                imageContainer.className = 'image-container';
                
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'image-preview';
                img.alt = '새 이미지';
                
                const removeBtn = document.createElement('button');
                removeBtn.type = 'button';
                removeBtn.className = 'remove-btn';
                removeBtn.innerHTML = '×';
                removeBtn.addEventListener('click', function() {
                    const index = selectedFiles.indexOf(file);
                    if (index > -1) {
                        selectedFiles.splice(index, 1);
                    }
                    imageContainer.remove();
                });
                
                imageContainer.appendChild(img);
                imageContainer.appendChild(removeBtn);
                previewContainer.appendChild(imageContainer);
            };
            reader.readAsDataURL(file);
        });
        
        // input 초기화
        this.value = '';
    });

    // 기존 이미지 삭제 버튼 처리
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('existing-remove')) {
            const imagePath = e.target.getAttribute('data-path');
            removedExistingImages.push(imagePath);
            e.target.parentElement.remove();
        }
    });

    // 저장 버튼 클릭 처리
    saveBtn.addEventListener('click', function() {
        const productId = document.getElementById('productId').value;
        
        // 상세정보 가져오기
        const detailInfo = document.getElementById('detailInfo').value.trim();

        // 필수 필드 검증
        if (!detailInfo) {
            alert('상품 상세정보를 입력해주세요.');
            return;
        }

        const formData = new FormData();
        formData.append('productId', productId);
        formData.append('detailInfo', detailInfo);

        // 새로운 파일들 추가 (인테리어 게시판 방식)
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        // 삭제된 이미지 경로들 추가
        removedExistingImages.forEach(path => {
            formData.append('removedImages', path);
        });

        fetch('/admin/product-details/save', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('상품 상세정보가 저장되었습니다.');
                window.location.href = '/admin/product-details';
            } else {
                alert('저장 중 오류가 발생했습니다: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('저장 중 오류가 발생했습니다.');
        });
    });

    // 취소 버튼
    cancelBtn.addEventListener('click', function() {
        if (confirm('작성 중인 내용이 삭제됩니다. 정말 취소하시겠습니까?')) {
            window.location.href = '/admin/product-details';
        }
    });
}); 