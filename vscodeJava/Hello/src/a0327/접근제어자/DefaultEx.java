package a0327.접근제어자;

class DefaultClass {
    void display() { // default가 앞에 생략된 형태
        System.out.println("package(폴더) 내부에서 접근 가능");
    }
}

public class DefaultEx {
    public static void main(String[] args) {
        DefaultClass obj = new DefaultClass();
        obj.display(); // 정상 실행 (같은 패키지)
    }
}
