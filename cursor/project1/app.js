// 애플리케이션 메인 JavaScript 파일

// 전역 변수
let applications = [];

// DOM 요소들
const modal = document.getElementById('applicationModal');
const closeBtn = document.querySelector('.close');
const applicationForm = document.getElementById('applicationForm');
const adminBtn = document.getElementById('adminBtn');

// Password modal elements
const passwordModal = document.getElementById('passwordModal');
const adminPassword = document.getElementById('adminPassword');
const passwordSubmit = document.getElementById('passwordSubmit');
const passwordCancel = document.getElementById('passwordCancel');
const passwordError = document.getElementById('passwordError');

// Admin password
const ADMIN_PASSWORD = '123';



// 폼 제출 처리
async function handleFormSubmission(formData) {
    const name = formData.get('name');
    const email = formData.get('email');
    const phone = formData.get('phone');
    const motivation = formData.get('motivation');
    
    try {
        // Firebase에 데이터 저장
        await db.collection('applications').add({
            name: name,
            email: email,
            phone: phone,
            motivation: motivation,
            timestamp: firebase.firestore.FieldValue.serverTimestamp()
        });
        
        // 성공 메시지 표시
        alert(`신청이 완료되었습니다!\n\n이름: ${name}\n이메일: ${email}\n휴대폰: ${phone}\n\n곧 연락드리겠습니다.`);
        
        // 모달 닫기 및 폼 초기화
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
        applicationForm.reset();
        
    } catch (error) {
        console.error('저장 오류:', error);
        alert('신청 저장 중 오류가 발생했습니다. 다시 시도해주세요.');
    }
}

// 비밀번호 모달 열기
function openPasswordModal() {
    passwordModal.style.display = 'block';
    document.body.style.overflow = 'hidden';
    adminPassword.focus();
    passwordError.style.display = 'none';
    adminPassword.value = '';
}

// 비밀번호 모달 닫기
function closePasswordModal() {
    passwordModal.style.display = 'none';
    document.body.style.overflow = 'auto';
    adminPassword.value = '';
    passwordError.style.display = 'none';
}

// 비밀번호 확인
function checkPassword() {
    const password = adminPassword.value;
    
    if (password === ADMIN_PASSWORD) {
        closePasswordModal();
        // 새로운 창에서 관리자 대시보드 열기
        window.open('admin.html', '_blank', 'width=1400,height=800,scrollbars=yes,resizable=yes');
    } else {
        passwordError.style.display = 'block';
        adminPassword.value = '';
        adminPassword.focus();
    }
}

// 이벤트 리스너 설정
document.addEventListener('DOMContentLoaded', function() {
    // 모달 열기 (CTA 버튼 클릭)
    document.querySelectorAll('.cta-button').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        });
    });

    // 모달 닫기 (X 버튼)
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    });

    // 모달 닫기 (배경 클릭)
    window.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    });

    // 관리자 버튼 클릭
    adminBtn.addEventListener('click', openPasswordModal);

    // 비밀번호 모달 이벤트
    passwordSubmit.addEventListener('click', checkPassword);
    passwordCancel.addEventListener('click', closePasswordModal);
    
    // 비밀번호 입력에서 Enter 키 처리
    adminPassword.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            checkPassword();
        }
    });
    
    // 비밀번호 모달 배경 클릭으로 닫기
    window.addEventListener('click', function(e) {
        if (e.target === passwordModal) {
            closePasswordModal();
        }
    });



    // 폼 제출 처리
    applicationForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(applicationForm);
        
        // 폼 검증
        const name = formData.get('name');
        const email = formData.get('email');
        const phone = formData.get('phone');
        const motivation = formData.get('motivation');
        
        if (!name || !email || !phone || !motivation) {
            alert('모든 필수 항목을 입력해주세요.');
            return;
        }
        
        // 이메일 검증
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            alert('올바른 이메일 주소를 입력해주세요.');
            return;
        }
        
        // 휴대폰 검증
        const phoneRegex = /^[0-9-]+$/;
        if (!phoneRegex.test(phone)) {
            alert('올바른 휴대폰 번호를 입력해주세요.');
            return;
        }
        
        // Firebase에 데이터 저장
        handleFormSubmission(formData);
    });

    // 네비게이션 스무스 스크롤
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // 헤더 스크롤 효과
    window.addEventListener('scroll', function() {
        const header = document.querySelector('header');
        if (window.scrollY > 100) {
            header.style.background = 'rgba(255, 255, 255, 0.98)';
        } else {
            header.style.background = 'rgba(255, 255, 255, 0.95)';
        }
    });
}); 