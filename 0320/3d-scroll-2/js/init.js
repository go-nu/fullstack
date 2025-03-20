$(function(){
    //window 스크롤시 
    // scrolTop 값을 구해서 sct 값으로 저장
    //각각의 article에 transform : translate값을
    // sct 값으로 넣어서
    //축구선수들이 스크롤 할때마다 점점가까이 오게 하시오

    $(window).scroll(function(){
        var sct = $(this).scrollTop();
        // $('section>article').eq(0).css({'transform':'translateZ('+sct+'px)'});
        // $('section>article').eq(1).css({'transform':'translateZ('+((-5000)+sct)+'px)'});
        // $('section>article').eq(2).css({'transform':'translateZ('+((-10000)+sct)+'px)'});
        // $('section>article').eq(3).css({'transform':'translateZ('+((-15000)+sct)+'px)'});
        // $('section>article').eq(4).css({'transform':'translateZ('+((-20000)+sct)+'px)'});
    // for(var i=0; i < 5;i++){
    //     $('section>article').eq(i).css({'transform':'translateZ('+((-5000*i)+sct)+'px)'}); 

    //         if(sct >= i*5000 && sct < (i+1)*5000){
    //         $('article').removeClass('on');
    //         $('article').eq(i).addClass('on');
    //         $('.scrollNav li').removeClass('on');
    //         $('.scrollNav li').eq(i).addClass('on');
    //     }
    // }

    // if(sct >=0 && sct <5000){
    //     $('article').removeClass('on');
    //     $('article').eq(0).addClass('on');
    //     $('.scrollNav li').removeClass('on');
    //     $('.scrollNav li').eq(0).addClass('on');
    // }
    // if(sct >=5000 && sct <10000){
    //     $('article').removeClass('on');
    //     $('article').eq(1).addClass('on');
    //     $('.scrollNav li').removeClass('on');
    //     $('.scrollNav li').eq(1).addClass('on');
    // }
    // if(sct  >=10000  && sct < 15000){
    //     $('article').removeClass('on');
    //     $('article').eq(2).addClass('on');
    //     $('.scrollNav li').removeClass('on');
    //     $('.scrollNav li').eq(2).addClass('on');
    //   }
    //   if(sct >=15000 && sct < 20000 ){
    //         $('article').removeClass('on');
    //         $('article').eq(3).addClass('on');
    //         $('.scrollNav li').removeClass('on');
    //         $('.scrollNav li').eq(3).addClass('on');
    //     }	
    // if(sct >=20000 && sct < 25000 ){
    //     $('article').removeClass('on');
    //     $('article').eq(4).addClass('on');
    //     $('.scrollNav li').removeClass('on');
    //     $('.scrollNav li').eq(4).addClass('on');
    //   }


        $('section>article').each(function(i){
            $('section>article').eq(i).css({'transform':'translateZ('+((-5000*i)+sct)+'px)'}); 
            if(sct >= i*5000 && sct < (i+1)*5000){
                $('article').removeClass('on');
                $('article').eq(i).addClass('on');
                $('.scrollNav li').removeClass('on');
                $('.scrollNav li').eq(i).addClass('on');
            }
        });
    });

    $('.scrollNav li').click(function(){
        var i = $(this).index();
        $('html,body').stop().animate({scrollTop:5000*i},1500,'swing');
    });

    $('body').on('mousemove',function(e){

        var posX = e.pageX/100;
        var posY = e.pageY/150;
        $( '.obj11' ).css( { 
            'left': -270 - posX, 'bottom': -100 - posY 
        } );
        $( '.obj12' ).css( { 
            'right': -590 - posX, 'top': -90 + posY 
        } );
        $( '.obj13' ).css( { 
            'left': -100 + posX, 'bottom': 20 + posY 
        } );

        /*2페이지*/
        $( '.obj21' ).css( { 
            'right': -710 - posX, 'bottom': -420 - posY 
        } );
        $( '.obj22' ).css( { 
            'right': -50 + posX, 'bottom': -100 + posY 
        } );

        /*3페이지*/
        $( '.obj31' ).css( { 
            'right': 110 - posX, 'bottom': -70 - posY 
        } );
        $( '.obj32' ).css( { 
            'left': 100 - posX, 'bottom': -160 - posY 
        } );

        /*4페이지*/
        $( '.obj41' ).css( { 
            'left': 350 + posX, 'bottom': -150 - posY 
        } );
        $( '.obj42' ).css( { 
            'right': 200 + posX, 'top': -250 - posY 
        } );
        $( '.obj43' ).css( { 
            'right': -100 + posX, 'bottom': -120 + posY 
        } );

        /*5페이지*/
        $( '.obj51' ).css( { 
            'left': -100 - posX, 'bottom': -200 - posY 
        } );
        $( '.obj52' ).css( { 
            'right': 320 + posX, 'top': -50 - posY 
        } );
        $( '.obj53' ).css( { 
            'left': -30 + posX, 'bottom': 0 - posY 
        } );

    });

    
});