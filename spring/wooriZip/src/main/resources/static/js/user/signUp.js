document.addEventListener("DOMContentLoaded", () => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("success") === "true") {
        Swal.fire({
            title: "회원가입 완료!",
            text: "로그인 화면으로 이동해 주세요.",
            icon: "success",
            confirmButtonText: "확인"
        });
    }
});

let isEmailChecked = false;

document.addEventListener('DOMContentLoaded', function () {
    const emailInput = document.getElementById('email');
    const signupBtn = document.getElementById('signupBtn');

    // 이메일 입력값 변경 시 중복 확인 상태 초기화
    emailInput.addEventListener('input', function () {
        isEmailChecked = false;
        document.getElementById('emailCheckMsg').textContent = '';
        if (signupBtn) {
            signupBtn.disabled = true;
        }
    });
});

function checkEmailDuplicate() {
    const emailId = document.getElementById('email-id').value.trim();
    const emailDomain = document.getElementById('email-domain').value.trim();
    const signupBtn = document.getElementById('signupBtn');

    if (emailId === '' || emailDomain === '') {
        alert('이메일을 입력해 주세요.');
        return;
    }

    // 이메일 아이디와 도메인이 모두 입력되었을 때만 중복 체크
    if (emailId && emailDomain) {
        const fullEmail = emailId + '@' + emailDomain; // 전체 이메일 생성

        // 이메일 중복 확인을 서버에 요청
        const msgEl = document.getElementById('emailCheckMsg');
        const xhr = new XMLHttpRequest();
        xhr.open('GET', '/user/checkEmail?email=' + encodeURIComponent(fullEmail), true);
        xhr.onload = function () {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                if (response.exists) {
                    msgEl.textContent = '이미 사용 중인 이메일입니다.';
                    msgEl.style.color = 'red';
                    signupBtn.disabled = true;
                } else {
                    msgEl.textContent = '사용 가능한 이메일입니다.';
                    msgEl.style.color = 'green';
                    signupBtn.disabled = false;

                    // 이메일을 hidden input에 설정
                    document.getElementById('email').value = fullEmail;
                }
            } else {
                msgEl.textContent = '오류가 발생했습니다. 다시 시도해 주세요.';
                msgEl.style.color = 'red';
                signupBtn.disabled = true;
            }
        };
        xhr.onerror = function () {
            msgEl.textContent = '서버 통신 중 오류가 발생했습니다.';
            msgEl.style.color = 'red';
            signupBtn.disabled = true;
        };
        xhr.send();
    } else {
        // 이메일이 완성되지 않으면 버튼을 비활성화
        signupBtn.disabled = true;
    }
}

document.addEventListener("DOMContentLoaded", function () {
    const selectGroups = document.querySelectorAll(".select-buttons");

    selectGroups.forEach(group => {
        const labels = group.querySelectorAll("label");
        labels.forEach(label => {
            label.addEventListener("click", () => {
                labels.forEach(l => l.classList.remove("active"));
                label.classList.add("active");
            });
        });

        // 처음 체크된 radio의 label에 active 주기
        const checkedInput = group.querySelector("input:checked");
        if (checkedInput) {
            const checkedLabel = group.querySelector(`label[for="${checkedInput.id}"]`);
            if (checkedLabel) checkedLabel.classList.add("active");
        }
    });
});

// 비밀번호 확인
document.addEventListener('DOMContentLoaded', function () {
    var userPwInput = document.getElementById('userPw');
    var checkPwInput = document.getElementById('checkPw');
    var pwCheckResult = document.getElementById('pwCheckResult');

    function checkPasswordsMatch() {
        if (userPwInput.value === '' && checkPwInput.value === '') {
            pwCheckResult.textContent = '';
            pwCheckResult.className = '';
            return;
        }

        if (userPwInput.value === checkPwInput.value) {
            pwCheckResult.textContent = '비밀번호가 일치합니다.';
            pwCheckResult.className = 'success';
        } else {
            pwCheckResult.textContent = '비밀번호가 일치하지 않습니다. 다시 입력해 주세요.';
            pwCheckResult.className = 'error';
        }
    }

    userPwInput.addEventListener('input', checkPasswordsMatch);
    checkPwInput.addEventListener('input', checkPasswordsMatch);
});

// 전화번호 합치기
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

// 주소 API
function sample4_execDaumPostcode() {
    new daum.Postcode({
        oncomplete: function(data) {
            var roadAddr = data.roadAddress;
            var extraRoadAddr = '';

            if(data.bname !== '' && /[동|로|가]$/g.test(data.bname)){
                extraRoadAddr += data.bname;
            }
            if(data.buildingName !== '' && data.apartment === 'Y'){
                extraRoadAddr += (extraRoadAddr !== '' ? ', ' + data.buildingName : data.buildingName);
            }
            if(extraRoadAddr !== ''){
                extraRoadAddr = ' (' + extraRoadAddr + ')';
            }

            document.getElementById("sample4_postcode").value = data.zonecode;
            document.getElementById("sample4_roadAddress").value = roadAddr;
            document.getElementById("sample4_jibunAddress").value = data.jibunAddress;

            if(roadAddr !== ''){
                document.getElementById("sample4_extraAddress").value = extraRoadAddr;
            } else {
                document.getElementById("sample4_extraAddress").value = '';
            }

            var guideTextBox = document.getElementById("guide");
            if(data.autoRoadAddress) {
                var expRoadAddr = data.autoRoadAddress + extraRoadAddr;
                guideTextBox.innerHTML = '(예상 도로명 주소 : ' + expRoadAddr + ')';
                guideTextBox.style.display = 'block';
            } else if(data.autoJibunAddress) {
                var expJibunAddr = data.autoJibunAddress;
                guideTextBox.innerHTML = '(예상 지번 주소 : ' + expJibunAddr + ')';
                guideTextBox.style.display = 'block';
            } else {
                guideTextBox.innerHTML = '';
                guideTextBox.style.display = 'none';
            }
        }
    }).open();
}

//이메일 입력
document.addEventListener("DOMContentLoaded", function () {
    const domainInput = document.getElementById("email-domain");
    const domainSelect = document.getElementById("domain-select");
    const emailIdInput = document.getElementById("email-id");
    const signupBtn = document.getElementById("signupBtn");

    // select 선택 시 → input 값을 해당 도메인으로 변경
    domainSelect.addEventListener("change", function () {
        if (this.value !== "") {
            domainInput.value = this.value;
        } else {
            domainInput.value = "";
        }
        validateEmail();
    });

    // input 수동 입력 시 select 값을 "직접입력"으로 되돌림
    domainInput.addEventListener("input", function () {
        domainSelect.value = "";
        validateEmail();
    });

    function validateEmail() {
        const emailId = emailIdInput.value.trim();
        const emailDomain = domainInput.value.trim();
        const email = emailId + "@" + emailDomain;

        if (emailId && emailDomain) {
            checkEmailDuplicate(email);
        } else {
            signupBtn.disabled = true;
        }
    }
});