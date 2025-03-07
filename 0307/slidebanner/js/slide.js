$(function(){
    var visual = $('#brandVisual > ul > li');
    var button = $('#buttonList > li');
    var current = 0;
    var id;
    var i = 0;

    button.click(function(){
        i = $(this).index();
        // alert(i);
        button.removeClass('on');
        button.eq(i).addClass('on');
        move();
        return false;
    });

    // timer();
    function timer(){
        id = setInterval(function(){
            var n = current + 1;
            console.log(n);
            if(n == 3) n = 0;
            button.eq(n).trigger('click');
            // n번째 버튼을 3초마다 클릭
        }, 3000);
    }
    
    function move(){
        if(current == i) return
        // 현재 활성화된 버튼 = 클릭한 버튼이면 나가기.
        var cu = visual.eq(current); // 현재 사진
        var ne = visual.eq(i); // 슬라이드에 들어올 사진
        cu.css('left','0').stop().animate({'left':'-100%'},500);
        ne.css('left','100%').stop().animate({'left':'0'},500);
        current = i;
    }






});