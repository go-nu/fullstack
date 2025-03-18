addEventListener('resize', function () {
    var contents = document.querySelectorAll('.content');
    for (var _i = 0, contents_1 = contents; _i < contents_1.length; _i++) {
        var item = contents_1[_i];
        item.style.height = innerHeight + "px";
    }
});
var e = document.createEvent('Event');
e.initEvent('resize', true, true);
dispatchEvent(e);
var content = document.querySelector('#scene-2-content'); //챠트전체를 content
var charts = content.querySelectorAll('.chart'); //각각의 챠트
var percentDatas = [];
for (var _i = 0, charts_1 = charts; _i < charts_1.length; _i++) {
    var item = charts_1[_i];
    percentDatas.push(item.querySelector('.percent-number').innerText);
    // percentData 입력값들을 배열에 저장
}
addEventListener('scroll', function () {
    if (pageYOffset > 1200) {
        zeroPercent();
    }
    if (600 <= pageYOffset && pageYOffset < 1200) {
        activateScene2();
    }
});
function activateScene2() {
    if (!(content.classList.contains('active'))) {
        content.style.transition = 'right 1.2s';
        content.style.right = '0px';
        var _loop_1 = function (i) {
            var chart = charts[i];
            var circleLeft = chart.querySelector('.left .circle-mask-inner');
            circleLeft.style.transform = 'rotate(0)';
            var circleRight = chart.querySelector('.right .circle-mask-inner');
            circleRight.style.transform = 'rotate(0)'; //원형챠트 회전 0
            var percentNumber = chart.querySelector('.percent-number');
            var percentData = percentDatas[i];
            percentNumber.innerText = "0";
            setTimeout(percent, 1000);
            function percent() {
                var k = -90;
                function percentAnimate() {
                    if (k > 90) {
                        k = 90;
                    }
                    var easeUp = ((Math.sin(k * Math.PI / 180)) + 1) / 2;
                    var now = percentData * easeUp;
                    var deg = now * 360 / 100;
                    var degRight = Math.min(Math.max(deg, 0), 180);
                    var degLeft = Math.min(Math.max(deg - 180, 0), 180);
                    circleRight.style.transform = "rotate(" + degRight + "deg)";
                    circleLeft.style.transform = "rotate(" + degLeft + "deg)";
                    percentNumber.innerText = "" + Math.floor(now);
                    //글씨가 0 부터 now까지 증가
                    k += 180 / (60 * 1.5);
                    if (k > 90) {
                        return clearInterval(id);
                    }
                }
                var id = setInterval(percentAnimate, 1000 / 60);
            }
        };
        for (var i = 0; i < charts.length; i++) {
            _loop_1(i);
        }
    }
    content.classList.add('active');
    content.classList.remove('unactive');
}
function zeroPercent() {
    if (!(content.classList.contains('unactive'))) {
        var circleLR = document.querySelectorAll('.left .circle-mask-inner, .right .circle-mask-inner');
        for (var _i = 0, circleLR_1 = circleLR; _i < circleLR_1.length; _i++) {
            var item = circleLR_1[_i];
            item.style.transform = 'rotate(0)';
        }
    }
    content.classList.add('unactive');
    content.classList.remove('active');
}
