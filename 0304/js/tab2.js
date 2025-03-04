const tabs = document.querySelectorAll('.tab_menu li'); // tab button
const tabList = document.querySelectorAll('#tabContent > div'); // tab content

tabs.forEach(function(tab, i){ // function(element, index)
    tab.addEventListener('click', function(e){
        e.preventDefault(); // 링크 금지

        // .tab_menu li에 모든 li.active 삭제
        tabs.forEach(function(item){
            item.classList.remove("active");
        });

        tabs[i].classList.add("active");

        tabList.forEach(function(item){
            item.classList.remove("on");
        });

        tabList[i].classList.add("on");
    });
});
