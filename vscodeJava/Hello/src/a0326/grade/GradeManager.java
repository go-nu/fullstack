package a0326.grade;

import java.util.ArrayList;
import java.util.Scanner;

public class GradeManager {
    private static ArrayList<Student> students = new ArrayList<>();
    private static Scanner scanner = new Scanner(System.in);
    private static int idCounter = 1;

    public static void main(String[] args) {
        while (true) {
            System.out.println("\n=== 성적 관리 프로그램 ===");
            System.out.println("1. 성적 추가");
            System.out.println("2. 성적 조회");
            System.out.println("3. 성적 수정");
            System.out.println("4. 성적 삭제");
            System.out.println("5. 종료");
            System.out.print("선택: ");
            int choice = scanner.nextInt();
            scanner.nextLine();
            switch (choice) {
                case 1:
                    addStudent();
                    break;
                case 2:
                    viewStudent();
                    break;
                case 3:
                    updateStudent();
                    break;
                case 4:
                    deleteStudent();
                    break;
                case 5:
                    System.out.println("프로그램 종료.");
                    // return;
                    System.exit(0);
                default:
                    System.out.println("잘못된 입력입니다. 다시 선택하세요.");
                    break;
            }
        }
    }

    // 성적 추가
    private static void addStudent() {
        System.out.println("학생 이름 : ");
        String name = scanner.nextLine();
        System.out.println("점수 입력 : ");
        int score = scanner.nextInt();
        scanner.nextLine();

        // Student s = new Student();
        // s.setId(idCounter++);
        // s.setName(name);
        // s.setScore(score);
        Student student = new Student(idCounter++, name, score);
        students.add(student);
        System.out.println("성적이 추가되었습니다.");
    }

    private static void viewStudent() {
        if(students.isEmpty()) { // 리스트가 비어있으면(= 등록된 학생이 없으면)
            System.out.println("등록된 성적이 없으습니다.");
        } else {
            System.out.println("\n-----성적목록-----");
            for(Student s : students) {
                s.display(); // 직접 만든 출력 method
                System.out.println(s.toString()); // Override한 toString()을 이용
            }
        }
    }
    
    private static void updateStudent() {
        System.out.println("삭제할 학생의 ID 입력 : ");
        int id = scanner.nextInt();
        scanner.nextLine();

        for(Student student : students) {
            if(student.getId() == id) {
                System.out.println("새로운 점수 입력 : ");
                int newScore = scanner.nextInt();
                scanner.nextLine();
                student.setScore(newScore);
                System.out.println("성적이 수정되었습니다.");
                return; // return을 써주지 않으면 학생을 찾아 수정한 후에도 size만큼 loop가 실행
            }
        }
        System.out.println("해당 ID의 학생을 찾을 수 없습니다.");
    }

    private static void deleteStudent() {
        System.out.println("삭제할 학생의 ID 입력 : ");
        int id = scanner.nextInt();
        scanner.nextLine();

        for(Student s : students) {
            if (s.getId() == id) {
                students.remove(s);
                System.out.println("해당 학생이 삭제되었습니다.");
                return;
            }
        }
        System.out.println("해당 ID의 학생을 찾을 수 없습니다.");
    }

}
