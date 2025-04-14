package a0409.musicApp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

class LoginMenu extends AbstractMenu {
    private static final LoginMenu instance = new LoginMenu(null);
    public static LoginMenu getInstance() {
        return instance;
    }

    private static final String Login_Menu_Text = 
    "===================================\n" + 
    "시작 화면입니다. 메뉴를 선택해주세요\n" + 
    "1. 로그인\t2. 회원가입\t3. 관리자 메뉴\t0.종료\n" + 
    "===================================\n" +
    "선택 >>";

    private LoginMenu(Menu prevMenu) {
        super(Login_Menu_Text, prevMenu);
    }

    @Override
    public Menu next() {
        defaultSong();
        System.out.println("현재 사용자 수: " + accounts.size());
        int ls = sc.nextInt();
        sc.nextLine();

        switch (ls) {
            case 1:
            String id = logIn();
            if (id != null) {
                setUser(id);
                MainMenu mm = MainMenu.getInstance();
                mm.setPrevMenu(this);
                mm.setUser(id);
                return mm;
            }
            return this;
            case 2:
                signIn();
                return this;
            case 3:
                changeAdmin();
                AdminMenu am = AdminMenu.getInstance();
                am.setPrevMenu(this);
                return am;
            case 0:
                return null;
            default:
                System.out.println("잘못된 입력입니다.");
                return this;
        }
    }

    public void defaultSong() {
        if (accounts.size() < 1) {
            accounts.add(new User("기본계정", 20000101, "abc", "123"));
        }
        try {
            File addfile = new File(".\\src\\a0409\\musicApp\\defaultPlayList.txt");
            BufferedReader br = new BufferedReader(new FileReader(addfile));
            String line;
            while ((line = br.readLine()) != null) {
                String[] s = line.split(",");
                aPlayList.add(new Song(s[0], s[1], s[2]));
            }
            br.close();

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        showAll();
    }

    User admin = new User("관리자", 950509, "admin123", "123*");

    private String logIn() {
        String id;
        String pw;
        System.out.println("로그인 화면입니다.");
        while (true) {
            System.out.print("ID : ");
            id = sc.nextLine();
            if (accounts.isEmpty()) {
                System.out.println("존재하는 계정이 없습니다. 회원가입을 먼저 진행해주세요.");
                return null;
            }
            for (User account : accounts) {
                if (id.equals(account.getId())) {
                    while (true) {
                        System.out.print("PW : ");
                        pw = sc.nextLine();
                        if (pw.equals(account.getPw())) {
                            System.out.println("로그인 성공, 메인 메뉴로 넘어갑니다.");
                            return id;
                        } else {
                            System.out.println("잘못된 비밀번호입니다. 다시 입력해주세요.");
                        }
                    }
                }
            }
            System.out.println("해당 ID는 존재하지 않습니다. 다시 시도해주세요.");
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
        while (true) {
            System.out.print("생년월일(8자리) : ");
            birth = sc.nextInt();
            sc.nextLine();
            int temp = birth;
            int count = 0;
            while (temp > 0) {
                temp /= 10;
                count++;
            }
            if(count == 8) break;
            else System.out.println("8자리를 입력하세요.");
        }
    
        while (true) {
            System.out.print("ID : ");
            id = sc.nextLine();
            if (accounts.isEmpty()) {
                System.out.print("비밀번호 : ");
                pw = sc.nextLine();
                accounts.add(new User(name, birth, id, pw));
                System.out.println("회원가입이 완료되었습니다!");
                break;
            }
    
            boolean sameID = false;
            for (User user : accounts) {
                if (user.getId().equals(id)) {
                    sameID = true;
                    break;
                }
            }
    
            if (sameID) {
                System.out.println("이미 존재하는 ID입니다. 다른 ID를 입력해주세요");
            } else {
                System.out.print("비밀번호 : ");
                pw = sc.nextLine();
                accounts.add(new User(name, birth, id, pw));
                System.out.println("회원가입이 완료되었습니다!");
                break;
            }
        }
    }
    
    

    private void changeAdmin() {
        String aID;
        String aPW;
        System.out.println("관리자 화면입니다.");
        Outter:while (true) {
            System.out.print("ID : ");
            aID = sc.nextLine();
            if (aID.equals(admin.getId())) {
                inner:while (true) {
                System.out.print("PW : ");
                aPW = sc.nextLine();
                    if(aPW.equals(admin.getPw())) {
                        System.out.println("관리자 인증 완료, 관리자 메뉴로 들어갑니다.");
                        break inner;
                    } else {
                        System.out.println("잘못된 비밀번호 입니다. 다시 입력해주세요");
                    }
                }
                break Outter;
            } else {
                System.out.println("관리자 계정이 아닙니다. 다시 입력해주세요.");
            }
        }
    }
}

