$(function(){
    setTimeout(function(){
        $('.slider li .text0').addClass('on');
        $('.slider li .atext0').addClass('on');

    }, 500);


    var bx = $('.slider').bxSlider({
        auto: true,
        controls: false,
        mode: 'fade',
        pager: false,
        pause: 5000, // 속도
        onSlideBefore: function(){},
        onSlideAfter: function(){
            var k = bx.getCurrentSlide(); // 현재 슬라이드 번호
            $('.slider li').find('h2').removeClass('on');
            $('.slider li').find('p').removeClass('on');
            $('.slider li .text' + k).addClass('on');
            $('.slider li .atext' + k).addClass('on');
        },
    })
});