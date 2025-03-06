$(function(){
    $('.sub').hide();
    $('.m_menu li ul').eq(0).show();
    $('.m_menu>li>a').click(function(){
        var kkk = $(this).next('.sub').css('display');
        // display 상태 반환 none, block
        // alert(kkk);
        if(kkk == 'none') {
            $('.sub').slideUp(); // 모든 슬라이드 닫기
            $(this).next('.sub').slideDown();
            // this의 .sub를 보여줌
        }

        return false;
    });
});