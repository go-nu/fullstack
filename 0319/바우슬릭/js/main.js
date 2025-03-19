$(function(){
    var pc_prev = $('.pc-slide .prev-btn');
    var pc_next = $('.pc-slide .next-btn');
    var mo_prev = $('.mo-slide .prev-btn');
    var mo_next = $('.mo-slide .next-btn');

    var $slick =  $('.pc-slide ul');
// pc용
  $slick.slick({
        autoplay:true,
        autoplaySpeed:3000,
        infinite:true,
        speed:1000,//현재슬라이드와 다음슬라이드 넘어가는 속도
        pauseOnHover:true,
        draggable:true,
        prevArrow:pc_prev,
        nextArrow:pc_next,
        dots:true,
    });
//mobile용
    $('.mo-slide ul').slick({
        autoplay:true,
        autoplaySpeed:4000,
        infinite:true,
        speed:1000,//현재슬라이드와 다음슬라이드 넘어가는 속도
        pauseOnHover:true,
        draggable:true,
        prevArrow:mo_prev,
        nextArrow:mo_next,
    });

    $(window).resize(function(){
        var w = $(this).width();
        if(w >= 800){
          mo_prev.hide();
          mo_next.hide();
        }else{
          mo_prev.show();
          mo_next.show();
        }
      });
      $(window).trigger("resize");

$slick.on('init',function(event, slick){

}); //slick 슬라이드 최초 실행할때 


$slick.on('afterChange',function(event, slick,currentSlide){
  //currentSlide 현재 바뀐 슬라이드 번호
  $('.num p strong').html(currentSlide + 1);
}); //slick 슬라이드 바뀔때

$('.play').click(function(){
  if($(this).hasClass('on')==false){
    $(this).addClass('on');
    $slick.slick('slickPause');
    //slick 슬라이드를 멈춤
  }else{
    $(this).removeClass('on');
    $slick.slick('slickPlay');
      //slick  슬라이드 재시작
  }
  return false;
});



});