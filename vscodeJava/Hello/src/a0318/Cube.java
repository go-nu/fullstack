package a0318;

public class Cube {
    public static void main(String[] args) {
        int n = 3;
        int v = volume(n);
        System.out.printf("한변의 길이가 %d인 정육면체의 부피는 %d", n, v);
    }
    public static int volume(int r) {
              
        return r * r * r;
    }
}
