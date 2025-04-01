package a0401.lambda;

interface Calculator2 {
    int sum1(int a, int b);
}

public class Function3 {
    public static void main(String[] args) {
        Calculator2 mc = Integer::sum; // Integer 요소를 합산해라. 기존 함수 sum
        int result = mc.sum1(3, 4); // 내가 만든 sum1 함수
        System.out.println(result);
    }
}
