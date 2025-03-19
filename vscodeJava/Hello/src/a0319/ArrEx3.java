package a0319;

public class ArrEx3 {
    public static void main(String[] args) {
        int[] numArr = {65, 74, 23, 75, 68, 96, 88, 98, 54};
        /*
        int max1 = 0;
        int max2 = 0;
        for(int i = 0; i < numArr.length; i++) {
            if(numArr[i] > max1) {
                max1 = numArr[i];
            }
        }
        for(int j = 0; j < numArr.length; j++) {
            if(numArr[j] == max1) continue;
            if(numArr[j] > max2) {
                max2 = numArr[j];
            }
        }
        
             */
        int secondTopIdx = getSecondTopIdx(numArr);
        
        System.out.println("배열: [65, 74, 23, 75, 68, 96, 88, 98, 54]");
        System.out.printf("두 번째로 큰 수: %d", numArr[secondTopIdx]);

    }
    private static int getSecondTopIdx(int[] arr) {
        int secondIdx = 0;
        int topIdx = getTopIndex(arr);
        for(int i=0; i < arr.length; i++){
            if(i == topIdx){
                continue; //최대값이 들어있는 인덱스번호일때 for문으로 복귀
            }
            if(arr[i] > arr[secondIdx]){
                topIdx = i;
            }
        }
        return secondIdx;
    }
    
    private static int getTopIndex(int[] arr) {
        int topIdx = 0; //최고값 인덱스번호 초기화
        for(int i=0; i < arr.length; i++){
            if(arr[i] > arr[topIdx]){
                topIdx = i;
            }
        }
        return topIdx; //최대값 인덱스 번호를 반환
    }
}
        



