$(function(){
    var slider = $('.slider').bxSlider({
        auto: true,
        controls: false,
        
    });

var tabs = $('.tab li');
var tabList = $('.sectionSlide > li > ul');
var current = 0;

tabs.click(function () {
    var i = $(this).index();
    tabs.find('a').removeClass('on');
    tabs.eq(i).find('a').addClass('on');
    move(i);
    return false;
});

function move(targetIndex) {
    if (current === targetIndex) return;

    var cu = tabList.eq(current); // 현재 사진
    var ne = tabList.eq(targetIndex); // 슬라이드에 들어올 사진

    cu.css('left', '0').stop().animate({ 'left': '-200%' }, 500);
    ne.css('left', '100%').stop().animate({ 'left': '0' }, 500);

    current = targetIndex;
}

$('.nextBtn').click(function (e) {
    e.preventDefault();
    var activeIndex = tabs.find('a.on').parent().index();

    if (activeIndex === 0) {
        var cl0 = tabList.eq(0);
        var cl1 = tabList.eq(4);

        if (parseInt(cl1.css('left')) !== 0) {
            cl0.css('left', '0').stop().animate({ 'left': '-100%' }, 500);
            cl1.css('left', '100%').stop().animate({ 'left': '0' }, 500);
        }
    } else if (activeIndex === 2) {
        var pr0 = tabList.eq(2);
        var pr1 = tabList.eq(5);

        if (parseInt(pr1.css('left')) !== 0) {
            pr0.css('left', '0').stop().animate({ 'left': '-100%' }, 500);
            pr1.css('left', '100%').stop().animate({ 'left': '0' }, 500);
        }
    }
});

$('.prevBtn').click(function (e) {
    e.preventDefault();
    var activeIndex = tabs.find('a.on').parent().index();

    if (activeIndex === 0) {
        var cl0 = tabList.eq(0);
        var cl1 = tabList.eq(4);

        if (parseInt(cl0.css('left')) !== 0) {
            cl1.css('left', '0').stop().animate({ 'left': '100%' }, 500);
            cl0.css('left', '-100%').stop().animate({ 'left': '0' }, 500);
        }
    } else if (activeIndex === 2) {
        var pr0 = tabList.eq(2);
        var pr1 = tabList.eq(5);

        if (parseInt(pr0.css('left')) !== 0) {
            pr1.css('left', '0').stop().animate({ 'left': '100%' }, 500);
            pr0.css('left', '-100%').stop().animate({ 'left': '0' }, 500);
        }
    }
});





    var ad = $('.ad_bxslider').bxSlider({
        auto:true,
        controls:false,

    })

    if($.cookie('popup') == 'none'){
        $('#notice_wrap').hide();
    }

    let chk = $('#expiresChk');
    $('.closeBtn').on('click',closePop);

    function closePop(){
        if(chk.is(":checked")){
            $.cookie('popup','none',{expires:1});
        }
        $('.popup').fadeOut("fast")
    }
});