let categoryTree = [];

// 페이지 로드 시 카테고리 트리 가져오기
window.onload = function () {
    fetch("/categories/tree")
    .then(res => res.json())
    .then(data => {
        categoryTree = data;
        populateParentCategory(data);
    });
};

// 대분류 드롭다운 채우기
function populateParentCategory(data) {
    const parentSelect = document.getElementById("parentCategory");
    parentSelect.innerHTML = `<option value="">-- 선택 --</option>`;
    data.forEach(cat => {
        parentSelect.innerHTML += `<option value="${cat.id}">${cat.name}</option>`;
    });

    parentSelect.addEventListener("change", function () {
        const selectedId = parseInt(this.value);
        const selected = categoryTree.find(c => c.id === selectedId);
        populateChildCategory(selected?.children || []);
    });
}

// 소분류 드롭다운 채우기
function populateChildCategory(children) {
    const childSelect = document.getElementById("childCategory");
    childSelect.innerHTML = `<option value="">-- 선택 --</option>`;
    children.forEach(child => {
        childSelect.innerHTML += `<option value="${child.id}">${child.name}</option>`;
    });
}

// 카테고리 기반 검색
function filterByCategory() {
    const categoryId = document.getElementById("childCategory").value || document.getElementById("parentCategory").value;
    if (!categoryId) {
        alert("카테고리를 선택해주세요.");
        return;
    }
    window.location.href = `/products?category=${categoryId}`;
}