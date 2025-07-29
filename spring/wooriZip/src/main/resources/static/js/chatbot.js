// 챗봇 JavaScript
let isTyping = false;

// 페이지 로드 시 현재 시간 표시
document.addEventListener('DOMContentLoaded', function() {
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);
});

// 현재 시간 업데이트
function updateCurrentTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit'
    });
    const timeElements = document.querySelectorAll('.message-time');
    if (timeElements.length > 0) {
        timeElements[timeElements.length - 1].textContent = timeString;
    }
}

// 키보드 이벤트 처리
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// 메시지 전송
function sendMessage(message = null) {
    const messageInput = document.getElementById('messageInput');
    const messageText = message || messageInput.value.trim();
    
    if (!messageText || isTyping) return;
    
    // 사용자 메시지 추가
    addMessage(messageText, 'user');
    
    // 입력창 초기화
    if (!message) {
        messageInput.value = '';
    }
    
    // 타이핑 표시
    showTypingIndicator();
    
    // 서버에 메시지 전송
    fetch('/api/chatbot/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'message=' + encodeURIComponent(messageText)
    })
    .then(response => response.json())
    .then(data => {
        hideTypingIndicator();
        handleBotResponse(data);
    })
    .catch(error => {
        console.error('Error:', error);
        hideTypingIndicator();
        addMessage('죄송합니다. 오류가 발생했습니다. 다시 시도해 주세요.', 'bot');
    });
}

// 봇 응답 처리
function handleBotResponse(response) {
    const { message, type, products, link, suggestions } = response;
    
    let responseContent = message;
    
    // 상품 목록이 있는 경우
    if (type === 'product_list' && products && products.length > 0) {
        responseContent += '<div class="product-list">';
        products.forEach(product => {
            responseContent += `
                <div class="product-card">
                    <h4>${product.name}</h4>
                    <p>${product.description || '상품 설명'}</p>
                    <p class="price">${formatPrice(product.price)}원</p>
                    <a href="/products/${product.id}" class="product-link">상품 보기</a>
                </div>
            `;
        });
        responseContent += '</div>';
    }
    
    // 링크가 있는 경우
    if (link) {
        let linkText = "상품 목록 보기";
        if (link === "/event") {
            linkText = "공지/이벤트 보기";
        }
        responseContent += `<br><a href="${link}" class="chat-link">${linkText}</a>`;
    }
    
    // 봇 메시지 추가
    addMessage(responseContent, 'bot');
    
    // 제안사항 업데이트
    if (suggestions && suggestions.length > 0) {
        updateSuggestions(suggestions);
    }
}

// 메시지 추가
function addMessage(content, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    messageDiv.innerHTML = `
        <div class="message-content">${content}</div>
        <div class="message-time">${timeString}</div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// 타이핑 표시
function showTypingIndicator() {
    if (isTyping) return;
    
    isTyping = true;
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typing-indicator';
    
    typingDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

// 타이핑 숨기기
function hideTypingIndicator() {
    isTyping = false;
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// 제안사항 업데이트
function updateSuggestions(suggestions) {
    const suggestionsContainer = document.getElementById('suggestions');
    suggestionsContainer.innerHTML = '';
    
    suggestions.forEach(suggestion => {
        const button = document.createElement('button');
        button.className = 'suggestion-btn';
        button.textContent = suggestion;
        button.onclick = () => sendMessage(suggestion);
        suggestionsContainer.appendChild(button);
    });
}

// 스크롤을 맨 아래로
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 가격 포맷팅
function formatPrice(price) {
    return price.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// 제안사항 버튼 클릭 이벤트
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('suggestion-btn')) {
        const message = e.target.textContent;
        sendMessage(message);
    }
}); 