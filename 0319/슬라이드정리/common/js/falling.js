/**
 * fallingObj 0.1
 * http://www.yamah.co.kr
 *
 * Copyright (C) 2016 yamah.co.kr
 */

(function($){
	$.fn.fallingObj = function(opt){
		
		var _wrap = this;
		var falling;
				
		// 초기 세팅
		opt = $.extend({
			maxLength : 10, // 갯수
			screenW : $(window).width(), // 추가 가로 범위
			screenH : $(window).height(), // 세로 범위(사라지는)
			closeBtn : _wrap.find('.btn_close'),
			inteval : 40,	
			mXmax : 150,	// x축 최대 속도 
			mXmin : 30,	// x축 최소 속도 
			mYmax : 3,	// y축 최대 속도 
			mYmin : 0.6,	// y축 최소 속도
			sizeMax : 30,	// 최대사이즈
			sizeMin : 10	// 최소 사이즈
		}, opt);
		
		_wrap.append('<ul class="falling"></ul>');
			
		for(var i=0;i< opt.maxLength; i++){
			$('.falling').append('<li></li>');
		}
		



		$('.falling > li').each(function(i){
			var $this =$(this);
			var angle = Math.floor((Math.random() * -200) + -20); //처음 시작 위치
			var posX = Math.floor((Math.random() * opt.screenW) + 0);
			var rdius = Math.floor((Math.random() * 10) + 10);
			var speed = Math.floor((Math.random() * opt.mXmax) + opt.mXmin);
			var tSpeed =((Math.random() * opt.mYmax) + opt.mYmin).toFixed(1);
			var size =Math.floor((Math.random() * opt.sizeMax) + opt.sizeMin);
			var opacityr = (Math.random() * 0.65) +0.4
			$(this).css({'opacity':opacityr.toFixed(1)});
			



			if(size >= 20){
				$(this).css({'z-index':'6'});
			}
			if(size >= 25){
				$(this).css({'z-index':'20'});
			}
			
			var rotate = function(){
				var posY = $this.position().top;
				var rnum = angle * (Math.PI/speed); 
				$this.css({
					'width':size,
					'height':size,
					'top':angle * tSpeed,
					'margin-left' :posX +  Math.cos(rnum) * rdius
				});
				if(posY <= opt.screenH){
					angle++;
				}else{
					angle = -10;
				};

			}
			var falling = setInterval(function(){
				rotate();
			}, opt.inteval);
				
			opt.closeBtn.click(function(){
				clearTimeout(falling);
				if(i >= opt.maxLength-1){
					winterclose()
				}
			});
		});
	}
})(jQuery);
//$('.falling').remove();