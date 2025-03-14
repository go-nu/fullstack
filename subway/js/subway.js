$(function(){
    var slider = $('.slider').bxSlider({
        auto: true,
        controls: false,
        
    });

    var tabs = $('.tab li')
    var tabList = $('.sectionSlide > ul')
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

        cu.css('left','0').stop().animate({'left':'-100%'},500);
        ne.css('left','100%').stop().animate({'left':'0'},500);
        current = i;
    }

    $('.nextBtn').click(function(){
        if(tabs.eq(0).find('a').hasClass('on')) {
            if(tabList.eq(0).find('.cl1').css('left') != '0px'){
                tabList.eq(0).find('.cl0').css('left','0').stop().animate({'left':'-100%'},500);
                tabList.eq(0).find('.cl1').css('left','100%').stop().animate({'left':'0'},500);
            }
        }
        if(tabs.eq(2).find('a').hasClass('on')) {
            if(tabList.eq(2).find('.pr1').css('left') != '0px'){
                tabList.eq(2).find('.pr0').css('left','0').stop().animate({'left':'-100%'},500);
                tabList.eq(2).find('.pr1').css('left','100%').stop().animate({'left':'0'},500);
            }
        }
    });
    
    $('.prevBtn').click(function(){
        if(tabs.eq(0).find('a').hasClass('on')) {
            if(tabList.eq(0).find('.cl0').css('left') != '0px'){
                tabList.eq(0).find('.cl1').css('left','0').stop().animate({'left':'100%'},500);
                tabList.eq(0).find('.cl0').css('left','-100%').stop().animate({'left':'0'},500);
            }
        }
        if(tabs.eq(2).find('a').hasClass('on')) {
            if(tabList.eq(2).find('.pr0').css('left') != '0px'){
                tabList.eq(2).find('.pr1').css('left','0').stop().animate({'left':'100%'},500);
                tabList.eq(2).find('.pr0').css('left','-100%').stop().animate({'left':'0'},500);
            }
        }
    });
    
});