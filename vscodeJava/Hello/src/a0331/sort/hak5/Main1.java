package a0331.sort.hak5;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;

public class Main1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ArrayList<Student> students = new ArrayList<>();

        // 학생 수 입력
        System.out.print("학생 수를 입력하세요: ");
        int n = sc.nextInt();
        sc.nextLine(); // 개행 문자 소비

        // 학생 정보 입력
        for (int i = 0; i < n; i++) {
            System.out.print("학생 이름: ");
            String name = sc.nextLine();
            System.out.print("학생 나이: ");
            int age = sc.nextInt();
            System.out.print("학생 학번: ");
            int studentId = sc.nextInt();
            sc.nextLine(); // 개행 문자 소비

            students.add(new Student(name, age, studentId));
        }

        // 정렬 옵션 선택
        System.out.println("정렬 기준을 선택하세요:");
        System.out.println("1. 이름");  
        System.out.println("2. 나이");  
        System.out.println("3. 학번");
        int choice = sc.nextInt();

        switch (choice) {
            case 1:
                students.sort(Comparator.comparing(Student::getName));
                break;
            case 2:
                students.sort(Comparator.comparingInt(Student::getAge));
                break;
            case 3:
                students.sort(Comparator.comparingInt(Student::getStudentId));
                break;    
            default:
                System.out.println("잘못된 선택입니다.");
                break;
        } 

        // 정렬된 결과 출력
        System.out.println("정렬된 학생 목록:");
        for (Student student : students) {
            System.out.println(student);
        }

        sc.close();
    }
}

// 학생 클래스 정의
class Student {
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
        return "이름: " + name + ", 나이: " + age + ", 학번: " + studentId;
    }
}