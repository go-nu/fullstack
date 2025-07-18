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