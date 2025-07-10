let selectedFiles = [];

document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("preview");

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      let files = Array.from(fileInput.files);

      if (files.length > 4) {
        alert("이미지는 최대 4장까지만 첨부할 수 있습니다.");
        files = files.slice(0, 4);
      }

      selectedFiles = files;
      const dataTransfer = new DataTransfer();
      preview.innerHTML = "";

      selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          const div = document.createElement("div");
          div.className = "position-relative me-2 mb-2";
          div.innerHTML = `
            <img src="${e.target.result}" style="height: 100px;" class="border rounded" />
            <button type="button" class="btn-close position-absolute top-0 end-0" onclick="removeImage(${index})"></button>
          `;
          preview.appendChild(div);
        };
        reader.readAsDataURL(file);
        dataTransfer.items.add(file);
      });

      fileInput.files = dataTransfer.files;
    });
  }
});

function removeImage(index) {
  selectedFiles.splice(index, 1);
  const dataTransfer = new DataTransfer();
  selectedFiles.forEach(f => dataTransfer.items.add(f));
  const fileInput = document.getElementById("fileInput");
  fileInput.files = dataTransfer.files;
  fileInput.dispatchEvent(new Event("change"));
}

function toggleEdit(id) {
  const form = document.getElementById("editForm-" + id);
  if (form) {
    form.classList.toggle("d-none");
  }
}

function toggleQnaForm() {
  const form = document.getElementById("qnaForm");
  if (form) {
    form.classList.toggle("d-none");
  } else {
    console.warn("QnA 작성 폼을 찾을 수 없습니다.");
  }
}
