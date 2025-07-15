// 카테고리 드롭다운 로딩
document.addEventListener("DOMContentLoaded", function () {
    fetch("/categories/parents")
        .then(res => res.json())
        .then(data => {
            const parentSelect = document.getElementById("parentCategory");
            data.forEach(c => {
                const option = document.createElement("option");
                option.value = c.id;
                option.text = c.name;
                parentSelect.appendChild(option);
            });
        });
});

function loadChildCategories() {
    const parentId = document.getElementById("parentCategory").value;
    const childSelect = document.getElementById("childCategory");
    childSelect.innerHTML = '<option value="">소분류 선택</option>';

    if (!parentId) return;

    fetch(`/categories/children?parentId=${parentId}`)
        .then(res => res.json())
        .then(data => {
            data.forEach(c => {
                const option = document.createElement("option");
                option.value = c.id;
                option.text = c.name;
                childSelect.appendChild(option);
            });
        });
}

// 이미지 미리보기 및 삭제 기능
const input = document.getElementById('imageInput');
const preview = document.getElementById('previewContainer');
const form = document.getElementById('productForm');
let selectedFiles = [];

// [수정 후: 항상 최신 파일만 반영]
input.addEventListener('change', function () {
    selectedFiles = Array.from(this.files);
    if (selectedFiles.length > 4) {
        alert("이미지는 최대 4장까지 업로드할 수 있습니다.");
        selectedFiles = selectedFiles.slice(0, 4);
        this.value = '';
        preview.innerHTML = '';
        return;
    }
    preview.innerHTML = '';
    selectedFiles.forEach(file => {
        const reader = new FileReader();
        reader.onload = e => {
            const box = document.createElement('div');
            box.className = 'img-box';
            const img = document.createElement('img');
            img.src = e.target.result;
            box.appendChild(img);
            preview.appendChild(box);
        };
        reader.readAsDataURL(file);
    });
});

// [폼 제출 시: 중복 append 제거]
form.addEventListener('submit', function (e) {
    e.preventDefault();
    const formData = new FormData(form);
    fetch(form.action, {
        method: 'POST',
        body: formData
    }).then(res => {
        if (res.redirected) {
            location.href = res.url;
        } else {
            alert("등록 실패");
        }
    }).catch(() => alert("에러 발생"));
});

// 옵션 동적 추가/삭제
function addOption() {
    const container = document.getElementById('optionContainer');
    const optionCount = container.children.length;
    if (optionCount >= 10) {
        alert("옵션은 최대 10개까지 추가할 수 있습니다.");
        return;
    }
    const div = document.createElement('div');
    div.className = 'option-item';
    div.innerHTML = `
        <label>모델명:</label>
        <select name="productModelDtoList[${optionCount}].productModelSelect" required>
            <option value="">옵션을 선택하세요</option>
            <option value="SUPER_SINGLE">슈퍼싱글</option>
            <option value="QUEEN">퀸</option>
            <option value="KING">킹</option>
            <option value="DEFAULT_MODEL">기본</option>
        </select><br/>
        <label>재고:</label>
        <input type="number" name="productModelDtoList[${optionCount}].prStock" placeholder="재고 입력"/><br/>
        <label>가격:</label>
        <input type="number" name="productModelDtoList[${optionCount}].price" placeholder="가격 입력"/><br/>
        <button type="button" class="removeOptionBtn">옵션 삭제</button>
        <hr />
    `;
    container.appendChild(div);
    updateRemoveButtons();
}

function updateRemoveButtons() {
    const container = document.getElementById('optionContainer');
    const items = container.querySelectorAll('.option-item');
    items.forEach((item, idx) => {
        const btn = item.querySelector('.removeOptionBtn');
        btn.style.display = (items.length > 1) ? '' : 'none';
        btn.onclick = function() {
            item.remove();
            // 옵션 삭제 후 name 인덱스 재정렬
            reorderOptionNames();
            updateRemoveButtons();
        };
    });
}

function reorderOptionNames() {
    const container = document.getElementById('optionContainer');
    const items = container.querySelectorAll('.option-item');
    items.forEach((item, idx) => {
        item.querySelector('select').name = `productModelDtoList[${idx}].productModelSelect`;
        item.querySelector('input[name$=".prStock"]').name = `productModelDtoList[${idx}].prStock`;
        item.querySelector('input[name$=".price"]').name = `productModelDtoList[${idx}].price`;
    });
}

// 최초 1개만 있을 때 삭제 버튼 숨김
document.addEventListener('DOMContentLoaded', function() {
    updateRemoveButtons();
});