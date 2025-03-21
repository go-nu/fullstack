package a0321;

public class StudentTest {
    public static void main(String[] args) {
        Student park = new Student(2019122104, "park");
        Student kim = new Student(2019112043, "kim");
        Student lee = new Student(2019152371, "lee");

        System.out.printf("Student 객체의 수: %d", Student.count);
    }
}
class Student {
    // 클래스 변수
    static int count = 0; // 학생수를 세기 위함
    // 인스턴스 변수
    int id;
    String name;
    // 생성자
    Student(int i, String n) {
        Student.count++; // 클래스변수에 객체가 생성될 때마다 ++
        id = i;
        name = n;
    }
}