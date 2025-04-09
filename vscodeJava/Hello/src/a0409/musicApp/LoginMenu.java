package a0409.musicApp;

import java.util.ArrayList;
import java.util.Scanner;

public class LoginMenu {
    Scanner sc = new Scanner(System.in);
    AdminMenu am = new AdminMenu();
    User admin = new User("관리자", 950509, "admin123", "123*");
    ArrayList<User> accounts;

    public void loginMenu() {
        while (true) {
            System.out.println("===================================");
            System.out.println("시작 화면입니다. 메뉴를 선택해주세요");
            System.out.println("1. 로그인\t2. 회원가입\t3. 관리자 메뉴\t0.종료");
            System.out.println("===================================");
            int ls = sc.nextInt();
            sc.nextLine();
            switch (ls) {
                case 1:
                    logIn();
                    break;
                case 2:
                    signIn();
                    break;
                case 3:
                    changeAdmin();
                case 0:
                    System.out.println("App 종료");
                default:
                    System.out.println("잘못된 입력입니다.");
                    break;
            }
            
        }
    }

    private void logIn() {
        String id;
        String pw;
        System.out.println("로그인 화면입니다.");
        System.out.print("ID : ");
        id = sc.nextLine();
        if (accounts.isEmpty() || accounts.size() == 0) {
            System.out.println("존재하는 계정이 없습니다. 회원가입을 먼저 진행해주세요.");
        } else {
            for(int i = 0; i < accounts.size(); i++) {
                if(id == accounts.get(i).getId()) {
                    System.out.print("PW : ");
                    pw = sc.nextLine();
                    if(pw == accounts.get(i).getPw()) {
                        // 로그인 성공 -> 사용자 메뉴
                    } else {
                        System.out.println("잘못된 비밀번호 입니다. 다시 입력해주세요");
                    }
                } else {
                    System.out.println("해당 ID는 존재하지 않습니다. 다시 시도해주세요");
                }
            }

        }
    }

    private void signIn() {
        String name;
        int birth;
        String id;
        String pw;

        System.out.println("회원가입 화면입니다.");
        System.out.print("이름 : ");
        name = sc.nextLine();
        System.out.print("생년월일(8자리) : ");
        birth = sc.nextInt();
        sc.nextLine();
        Outter:while (true) {
            System.out.print("ID : ");
            id = sc.nextLine();
            for(int i = 0; i < accounts.size(); i++) {
                if ((accounts.isEmpty() || accounts.size() == 0) || (accounts.get(i).getId() != id)){
                    System.out.print("비밀번호 : ");
                    pw = sc.nextLine();
                    accounts.add(new User(name, birth, id, pw));
                    // 회원가입 성공 -> 다시 메인메뉴로
                    break Outter;
                }
            }
            System.out.println("이미 존재하는 ID입니다. 다른 ID를 입력해주세요");
        }
    }

    private void changeAdmin() {
        String aID;
        String aPW;
        System.out.println("관리자 화면입니다.");
        System.out.print("관리자 ID : ");
        aID = sc.nextLine();
        if(aID == admin.getId()) {
            System.out.println("관리자 PW : ");
            aPW = sc.nextLine();
            if(aPW == admin.getPw()) {
                // 로그인 성공 -> 관리자 메뉴
            } else {
                System.out.println("잘못된 비밀번호 입니다. 다시 입력해주세요");
            }
        }
    }

}
