package a0318;

public class ArrEx2 {
    public static void main(String[] args) {
        int[] arrInt = new int[5]; // 길이가 5인 array 객체 생성 + 0으로 초기화

        System.out.println(arrInt[0]);
        for(int i = 0; i < arrInt.length; i++){
            System.out.println(arrInt[i]);
        }
        // 향상된 for 문
        // for(type 변수:배열이름)
        for(int num:arrInt) {
            System.out.println(num);
        }
    }
}
