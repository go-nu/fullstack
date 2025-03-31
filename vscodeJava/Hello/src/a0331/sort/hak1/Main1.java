package a0331.sort.hak1;

import java.util.Scanner;


public class Main1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // 학생수 입력
        System.out.print("학생 수를 입력하세요 : ");
        int n = sc.nextInt();
        sc.nextLine();

        // 학생 배열 생성
        Student[] students = new Student[n];
        // 학생 정보 입력
        for (int i = 0; i < n; i++) {
            System.out.print("\n학생 이름: ");
            String name = sc.nextLine();
            System.out.print("학생 나이: ");
            int age = sc.nextInt();
            System.out.print("학생 학번: ");
            int studentId = sc.nextInt();
            sc.nextLine(); // 개행 문자 소비
            

            students[i] = new Student(name, age, studentId);
        }
        
        // 삽입정렬
        InsertionSort(students);

        // 정렬된 결과 출력
        System.out.println("정렬된 학생 목록:");
        for (Student student : students) {
            System.out.println(student);
        }

        sc.close(); // Scanner 닫기
    }

    private static void InsertionSort(Student[] students) {
        int n = students.length;

        for(int i = 1; i < n; i++) {
            Student currentStudent = students[i];
            int j = i-1;
            while (j >= 0 && students[j].getName().compareTo(currentStudent.getName()) > 0) {
                // 오름차순
                students[j+1] = students[j];
                j--;
            }
            students[j+1] = currentStudent;
        }
    }
}

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
        return "Student [name=" + name + ", age=" + age + ", studentId=" + studentId + "]";
    }
}