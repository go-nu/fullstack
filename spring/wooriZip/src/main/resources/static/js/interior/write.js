window.addEventListener('scroll', function () {
    const header = document.querySelector('.header');
    if (window.scrollY > 10) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

let selectedFiles = [];

const input = document.getElementById('imageInput');
const preview = document.getElementById('previewContainer');

input.addEventListener('change', function () {
    const newFiles = Array.from(this.files);

    if (selectedFiles.length + newFiles.length > 4) {
        alert("이미지는 최대 4장까지 첨부할 수 있습니다.");
        return;
    }

    newFiles.forEach(file => {
        selectedFiles.push(file);

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

  this.value = '';
});

document.getElementById('submitBtn').addEventListener('click', function () {
    const form = document.getElementById('writeForm');
    const formData = new FormData(form);

    selectedFiles.forEach(file => {
        formData.append("files", file);
    });

    fetch('/interior/write', {
        method: 'POST',
        body: formData
    }).then(res => res.text())
    .then(result => {
        if (result === 'success') {
            alert("등록 완료!");
            location.href = '/interior';
        } else if (result === 'unauthorized') {
            alert("로그인 후 작성 가능합니다.");
            location.href = '/user/login';
        } else {
            alert("등록 실패");
        }
    });
});