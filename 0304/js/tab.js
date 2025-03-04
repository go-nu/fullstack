const targetLink = document.querySelectorAll('.tab_menu li a'); // tab button
const tabContent = document.querySelectorAll('#tabContent > div'); // tab content

targetLink.forEach(function(link){
    link.addEventListener('click', function(e){
        e.preventDefault(); // 링크 금지
        let orgTarget = e.target.getAttribute('href');
        // 클릭한놈(e.target) - this로 변경 가능
        // getAttribute는 href나 src 속성값 읽기
        // alert(orgTarget); #tabs1 #tabs2 #tabs3
        let tabTarget = orgTarget.replace('#', "");
        // #tabs1에서 앞에 붙은 #을 빈 문자열로 대체 -> tabs1
        // alert(tabTarget); tabs1 tabs2 tabs3

        tabContent.forEach(function(content){
            content.style.display = 'none';
        });
        // 클릭한 탭만 보이게 설정
        document.getElementById(tabTarget).style.display='block';
        targetLink.forEach(function(link2){
            link2.classList.remove('active');
        });
        e.target.classList.add('active');
    });
});
