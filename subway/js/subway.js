$(function(){
    var slider = $('.slider').bxSlider({
        auto: true,
        controls: false,
        
    });


    // const tabs = document.querySelectorAll('.tab li a');
    // const tabList = document.querySelectorAll('.sectionSlide ul');

    // tabs.forEach(function(tab, i){
    //     tab.addEventListener('click', function(e){
    //         e.preventDefault();

    //         tabs.forEach(function(item){
    //             item.classList.remove("on");
    //         });
    //         tabs[i].classList.add("on");

    //         tabList.forEach(function(item){
    //             item.classList.remove("active");
    //         });
    //         tabList[i].classList.add("active");
        
    //         move(i);
    //     });
    // });

    var tabs = $('.tab li')
    var tabList = $('.sectionSlide ul')
    var current = 0;
    var i = 0;
    
    tabs.click(function(){
        i = $(this).index();
        tabs.find('a').removeClass('on');
        tabs.eq(i).find('a').addClass('on');
        move();
        return false
    });

    function move(){
        if(current == i) return
        // 현재 활성화된 버튼 = 클릭한 버튼이면 나가기.
        var cu = tabList.eq(current); // 현재 사진
        var ne = tabList.eq(i); // 슬라이드에 들어올 사진
        // alert(tabList.index())

        cu.css('left','0').stop().animate({'left':'-100%'},500);
        ne.css('left','100%').stop().animate({'left':'0'},500);
        current = i;
    }
    if(tabs.eq(0).find('a').addClass('on')) {
        $('.nextBtn').click(function(){
            if(tabList.eq(0).css('left','0')){
                tabList.eq(0).css('left','0').stop().animate({'left':'-100%'},500);
                tabList.eq(1).css('left','100%').stop().animate({'left':'0'},500);
            } else {
                return
            }
        });
        $('.prevBtn').click(function(){
            if(tabList.eq(0).css('left','-100%')){
                tabList.eq(0).css('left','-100%').stop().animate({'left':'0'},500);
                tabList.eq(1).css('left','0').stop().animate({'left':'100%'},500);
            } else {
                return
            }
        });
    }


});