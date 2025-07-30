const email = /*[[${email}]]*/ null;
const error = /*[[${error}]]*/ null;

window.onload = function () {
    if (EMAIL !== null && EMAIL !== undefined) {
        showModal("회원님의 이메일은 " + EMAIL + " 입니다.");
    } else if (ERROR !== null && ERROR !== undefined) {
        showModal(ERROR);
    }
};

function showModal(message) {
    document.getElementById("resultMessage").innerText = message;
    document.getElementById("resultModal").classList.add("active");
}
function closeModal() {
    document.getElementById("resultModal").classList.remove("active");
}

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