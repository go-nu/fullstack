$(function () {
	$(window).resize(function(){
		var h=$(window).height();
		$(".content").height(h);
	});
	$(window).trigger("resize");
	var $content = $('#scene-2-content'); //챠트전체를 content
	var $charts = $content.find('.chart'); //각각의 챠트
	var percentDatas = [];

	$charts.each(function(){	
		percentDatas.push($(this).find(".percent-number").text());	
		// percentData 입력값들을 배열에 저장
	})

	$(window).scroll(function(){
		var sct = $(window).scrollTop();
		if(sct > 1200){
			zeroPercent();
		}
	    if(sct >=600 && sct <1200){
			activateScene2();
		}
	  }); 
	function activateScene2(){
		if($content.hasClass('active') ==false){
			$content.stop().animate({right:0},1200);
			$charts.each(function(i){
			var $chart = $(this);
			var $circleLeft = $chart.find('.left .circle-mask-inner')
			.css({transform:'rotate(0)'});
			var $circleRight = $chart.find('.right .circle-mask-inner')
			.css({transform:'rotate(0)'});	 //원형챠트 회전 0
				
			var $percentNumber = $chart.find(".percent-number");
			var percentData = $(percentDatas).get(i);	
			$percentNumber.text(0);	
			  $({percent:0}).delay(1000).animate({percent:percentData},{
			  	duration:1500,
				progress:function(){
				var now = this.percent;	
				var deg = now * 360 / 100;	
				
				var degRight = Math.min(Math.max(deg, 0), 180);
				var degLeft = Math.min(Math.max(deg - 180, 0), 180);
				$circleRight.css({transform:"rotate("+degRight+"deg)"});
				$circleLeft.css({transform:"rotate("+degLeft+"deg)"});
				$percentNumber.text(Math.floor(now));
				//글자가 0부터 now까지 증가
				}
			  });	
			});
		}	
	$content.addClass('active');	
	$content.removeClass('unactive');		
		
	}	
	
	function zeroPercent(){
		if($content.hasClass('unactive') ==false){
			var $chart = $(this);
			var $circleLeft = $chart.find('.left .circle-mask-inner')
			.css({transform:'rotate(0)'});
			var $circleRight = $chart.find('.right .circle-mask-inner')
			.css({transform:'rotate(0)'});	 //원형챠트 회전 0
				
			var $percentNumber = $chart.find(".percent-number");
			$percentNumber.text(0);
	
		}
		$content.addClass('unactive');	
		$content.removeClass('active');
	}

});






