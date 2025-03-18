package a0318;

import java.util.Arrays;

public class Arr3_1 {
    public static void main(String[] args) {
        int[] iArr1 = new int[10];
        int[] iArr2 = new int[10];
        int[] iArr3 = {100, 95, 80, 70, 60};
        char[] chArr = {'a', 'b', 'c', 'd', 'f'};
        // for문을 사용해 iArr1에 1,2,3,4,5,6,7,8,9,10 초기화
        for(int i = 0; i < iArr1.length; i++) {
            iArr1[i] = i+1;
        }
        System.out.println(Arrays.toString(iArr1));
        // iArr2에 1 ~ 10 무작위 정수 초기화
        for(int i = 0; i < iArr2.length; i++) {
            iArr2[i] = (int)(Math.random()*10)+1;
        }
        System.out.println(Arrays.toString(iArr2));

        System.out.println(iArr3); // 참조변수 (포인터? 데이터의 주소를 가르킴)
        System.out.println(chArr); // println method는 char 배열은 그대로 출력
    }
    
}
