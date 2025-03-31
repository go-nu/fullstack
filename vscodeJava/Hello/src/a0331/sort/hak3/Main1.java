package a0331.sort.hak3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;


public class Main1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ArrayList<Student> students = new ArrayList<>();

        // 학생수 입력
        System.out.print("학생 수를 입력하세요 : ");
        int n = sc.nextInt();
        sc.nextLine();

        // 학생 정보 입력
        for (int i = 0; i < n; i++) {
            System.out.print("\n학생 이름: ");
            String name = sc.nextLine();
            System.out.print("학생 나이: ");
            int age = sc.nextInt();
            System.out.print("학생 학번: ");
            int studentId = sc.nextInt();
            sc.nextLine(); // 개행 문자 소비
            

            students.add(new Student(name, age, studentId));
        }
        
        // 삽입정렬
        Collections.sort(students);

        // 정렬된 결과 출력
        System.out.println("정렬된 학생 목록:");
        for (Student student : students) {
            System.out.println(student);
        }

        sc.close(); // Scanner 닫기
    }

}

class Student implements Comparable<Student> {
    // Student 클래스가 Comparable<Student> 구현
    private String name;
    private int age;
    private int studentId;

    public Student(String name, int age, int studentId) {
        this.name = name;
        this.age = age;
        this.studentId = studentId;
    }

    public String getName() {
        return name;
    }
    public int getAge() {
        return age;
    }
    public int getStudentId() {
        return studentId;
    }

    @Override
    public String toString() {
        return "Student [name=" + name + ", age=" + age + ", studentId=" + studentId + "]";
    }

    @Override
    public int compareTo(Student o) {
        // 숫자비교
        return Integer.compare(this.age, o.age);
    }

    // element를 지정해 해당 element로 정렬
    // @Override
    // public int compareTo(Student o) {
            // 문자비교
    //     return this.name.compareTo(o.name);
    // }
}