package a0327.member1;

import java.util.Scanner;

public class Membership {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        MemberManager memberManager = new MemberManager();

        while (true) {
            System.out.println("\n===== 회원 관리 프로그램 =====");
            System.out.println("1. 회원 추가");
            System.out.println("2. 회원 검색");
            System.out.println("3. 회원 수정");
            System.out.println("4. 회원 삭제");
            System.out.println("5. 전체 회원 목록 보기");
            System.out.println("6. 종료");
            System.out.print("메뉴를 선택하세요: ");
            int choice = scanner.nextInt();
            scanner.nextLine(); // 버퍼 비우기

            switch (choice) {
                case 1:
                    System.out.println("회원 추가>");
                    System.out.print("추가할 회원의 이름 입력 : ");
                    String addName = scanner.nextLine();
                    System.out.print("추가할 회원의 나이 입력 : ");
                    int addAge = scanner.nextInt();
                    scanner.nextLine();
                    System.out.print("추가할 회원의 이메일 입력 : ");
                    String addEmail = scanner.nextLine();
                    memberManager.addMember(addName, addAge, addEmail);
                    System.out.println("회원 추가 완료.");
                    break;
                case 2:
                    System.out.print("검색할 회원 이름 입력 : ");
                    String searchName = scanner.nextLine();
                    Member foundMember = memberManager.findMember(searchName);
                    if (foundMember != null) {
                        System.out.println(foundMember);
                    } else {
                        System.out.println("해당 이름의 회원이 존재하지 않습니다.");
                    }
                    break;
                case 3:
                    System.out.print("수정할 회원의 이름을 입력하세요 : ");
                    String updateName = scanner.nextLine();
                    System.out.print("추가할 회원의 나이 입력 : ");
                    int newAge = scanner.nextInt();
                    scanner.nextLine();
                    System.out.print("추가할 회원의 이메일 입력 : ");
                    String newEmail = scanner.nextLine();
                    memberManager.updateMember(updateName, newAge, newEmail);
                    System.out.println("회원 정보가 수정되었습니다.");
                    
                    break;
                case 4:
                    System.out.println("삭제할 회원의 이름을 입력하세요 : ");
                    String delName = scanner.nextLine();
                    memberManager.delMember(delName);
                    break;
                case 5:
                    memberManager.displayAllMembers();
                    break;
                case 6:
                    System.out.println("프로그램 종료");
                    scanner.close();
                    System.exit(0);
                    break;
                default:
                    System.out.println("잘못된 입력입니다. 다시 입력하세요.");
                    break;
            }
        }
    }
}
