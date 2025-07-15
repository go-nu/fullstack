let selectedFiles = [];
let deleteIndexes = [];

function removeExistingImage(btn) {
    const container = btn.parentElement;
    const index = container.getAttribute("data-index");
    deleteIndexes.push(index);
    document.getElementById("deleteIndexes").value = deleteIndexes.join(",");
    container.remove();
}

document.getElementById("imageInput").addEventListener("change", function (event) {
    const previewArea = document.getElementById("previewArea");
    const newFiles = Array.from(event.target.files);

    if (selectedFiles.length + newFiles.length > 4) {
        alert("이미지는 최대 4장까지 업로드할 수 있습니다.");
        return;
    }

    newFiles.forEach(file => {
        selectedFiles.push(file);

        const reader = new FileReader();
        reader.onload = function (e) {
            const container = document.createElement('div');
            container.className = 'image-container';

            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'image-preview';

            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'remove-btn';
            btn.innerText = 'X';
            btn.onclick = () => {
                const idx = selectedFiles.indexOf(file);
                if (idx > -1) selectedFiles.splice(idx, 1);
                container.remove();
            };

            container.appendChild(img);
            container.appendChild(btn);
            previewArea.appendChild(container);
        };
        reader.readAsDataURL(file);
    });

    // input 초기화해서 같은 파일 다시 선택 가능하게
    event.target.value = '';
});

document.getElementById("editForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);

    selectedFiles.forEach(file => {
        formData.append("images", file); // 반드시 'images'로
    });

    fetch(`/products/${form.id.value}/edit`, {
        method: "POST",
        body: formData
    })
    .then(res => {
        if (!res.ok) throw new Error("서버 오류");
        return res.text();
    })
    .then(result => {
        alert("상품 수정 완료");
        window.location.href = `/products/${form.id.value}`;
    })
    .catch(err => {
        alert("수정 실패: " + err.message);
    });
});