package a0331.sort.hak4;

import java.util.ArrayList;
import java.util.Collections;
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

        // 학생 배열 생성
       // Student[] students = new Student[n];

        // 학생 정보 입력
        for (int i = 0; i < n; i++) {
            System.out.print("학생 이름: ");
            String name = sc.nextLine();
            System.out.print("학생 나이: ");
            int age = sc.nextInt();
            System.out.print("학생 학번: ");
            int studentId = sc.nextInt();
            sc.nextLine(); // 개행 문자 소비

            //students[i] = new Student(name, age, studentId);
            students.add(new Student(name, age, studentId));
        }
    //정렬 옵션 선택
    System.out.println("정렬 기준을 선택하세요");
    System.out.println("1. 이름");  
    System.out.println("2. 나이");  
    System.out.println("3. 학번");
    int choice = sc.nextInt();
    switch (choice) {
        case 1:
            //이름을 기준으로 정렬
            Collections.sort(students, new NameComparator());
            break;
        case 2:
            //이름을 기준으로 정렬
            Collections.sort(students, new AgeComparator());
            break;
        case 3:
            //이름을 기준으로 정렬
            Collections.sort(students, new StudentIdComparator());
            break;    
        default:
            break;
    } 



        // 정렬된 결과 출력
        System.out.println("정렬된 학생 목록:");
        for (Student student : students) {
            System.out.println(student);
        }

        sc.close(); // Scanner 닫기
    }

    
}

// 학생 클래스 정의
class Student {
    //Student 클래스가 Comparable<Student> 구현
    private String name;
    private int age;
    private int studentId;

    // 생성자
    public Student(String name, int age, int studentId) {
        this.name = name;
        this.age = age;
        this.studentId = studentId;
    }

    // Getter 메서드
    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public int getStudentId() {
        return studentId;
    }

   
    
}

class NameComparator implements Comparator<Student>{

    @Override
    public int compare(Student o1, Student o2) {
        return o1.getName().compareTo(o2.getName());
    }

}

class AgeComparator implements Comparator<Student>{

    @Override
    public int compare(Student o1, Student o2) {
        return Integer.compare(o1.getAge(), o2.getAge());
    }

}
class StudentIdComparator implements Comparator<Student>{

    @Override
    public int compare(Student o1, Student o2) {
        return Integer.compare(o1.getStudentId(), o2.getStudentId());
    }

}