package a0318;

public class Square {
    public static void main(String[] args) {
        int n = 4;
        int s = square(n);

        System.out.printf("한 변의 길이가 %d인 정사각형의 넓이 : %d", n, s);

    }
        
    public static int square(int length) {
        // 접근 제한자 - public : 누구든지 접근할 수 있다.
        // static : 객체 생성 없이 함수 호출
        int result = length * length;
        
        return result;
    }


}
