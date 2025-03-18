interface NodeListOf<TNode> extends Array<TNode> {}

addEventListener('resize',function(){
	const contents:NodeListOf<HTMLElement> = document.querySelectorAll('.content');
	for(const item of contents){
		item.style.height = `${innerHeight}px`;
	}
});

const e = document.createEvent('Event');
e.initEvent('resize',true,true);
dispatchEvent(e);

const content:HTMLElement = document.querySelector('#scene-2-content'); //챠트전체를 content
const charts = content.querySelectorAll('.chart'); //각각의 챠트
let percentDatas = [];

for(const item of charts){
	percentDatas.push((<HTMLElement>item.querySelector('.percent-number')).innerText)
	// percentData 입력값들을 배열에 저장
}


addEventListener('scroll',function(){
	if(pageYOffset > 1200){
		zeroPercent();
	}
	if(600 <= pageYOffset && pageYOffset < 1200){
		activateScene2();
	}
})

function activateScene2(){
	if(!(content.classList.contains('active'))){
		content.style.transition = 'right 1.2s';
		content.style.right = '0px';
		
		for(let i = 0 ; i < charts.length ; i ++){
			const chart = charts[i];
			const circleLeft:HTMLElement = chart.querySelector('.left .circle-mask-inner');
			circleLeft.style.transform = 'rotate(0)';
			const circleRight:HTMLElement = chart.querySelector('.right .circle-mask-inner');
			circleRight.style.transform = 'rotate(0)';	 //원형챠트 회전 0

			const percentNumber:HTMLElement = chart.querySelector('.percent-number');
			const percentData = percentDatas[i];
			percentNumber.innerText = `0`;

			setTimeout(percent,1000);

			function percent(){
				let k = -90;
			
				function percentAnimate() {
					if (k > 90) {
						k = 90;
					}
			
					const easeUp = ((Math.sin(k * Math.PI / 180)) + 1) / 2;
			
					const now = percentData*easeUp;
					const deg = now * 360 / 100;

					const degRight = Math.min(Math.max(deg, 0), 180);
					const degLeft = Math.min(Math.max(deg - 180, 0), 180);
					circleRight.style.transform = `rotate(${degRight}deg)`;
					circleLeft.style.transform = `rotate(${degLeft}deg)`;
					percentNumber.innerText = `${Math.floor(now)}`;
					//글씨가 0 부터 now까지 증가

					k += 180 / (60 * 1.5);
			
					if (k > 90) {
						return clearInterval(id);
					}
				}
				const id = setInterval(percentAnimate,1000/60);
			}
		}
	}
	content.classList.add('active');
	content.classList.remove('unactive');
}


function zeroPercent(){
	if(!(content.classList.contains('unactive'))){
		const circleLR:NodeListOf<HTMLElement> = document.querySelectorAll('.left .circle-mask-inner, .right .circle-mask-inner');
		for(const item of circleLR){
			item.style.transform = 'rotate(0)';
		}
	}
	content.classList.add('unactive');
	content.classList.remove('active');
}