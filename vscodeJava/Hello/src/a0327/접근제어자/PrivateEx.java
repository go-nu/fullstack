package a0327.접근제어자;

class Person {
    private String name; // 외부에서 직접 접근 불가

    public void setName(String name) { // public 메서드를 통해 접근
        this.name = name;
    }

    public String getName() {
        return name;
    }
}

public class PrivateEx {
    public static void main(String[] args) {
        Person p = new Person();
        // p.name = "John"; 오류, private은 직접 접근 불가
        p.setName("John");
        System.out.println(p.getName());

    }

}

