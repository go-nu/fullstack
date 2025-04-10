package a0409.musicApp;

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
        int ls = sc.nextInt();
        sc.nextLine();

        switch (ls) {
            case 1:
                logIn();
                MainMenu mm = MainMenu.getInstance();
                mm.setPrevMenu(this);
                return mm;
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

    User admin = new User("관리자", 950509, "admin123", "123*");

    private void logIn() {
        String id;
        String pw;
        System.out.println("로그인 화면입니다.");
        Outter:while (true) {
            System.out.print("ID : ");
            id = sc.nextLine();
            if (accounts.isEmpty() || accounts.size() == 0) {
                System.out.println("존재하는 계정이 없습니다. 회원가입을 먼저 진행해주세요.");
                break Outter;
            } else {
                for(int i = 0; i < accounts.size(); i++) {
                    if(id == accounts.get(i).getId()) {
                        inner:while (true) {
                        System.out.print("PW : ");
                        pw = sc.nextLine();
                            if(pw == accounts.get(i).getPw()) {
                                System.out.println("로그인 성공, 메인 메뉴로 넘어갑니다.");
                                break inner;
                            } else {
                                System.out.println("잘못된 비밀번호 입니다. 다시 입력해주세요");
                            }
                        }
                        break Outter;
                    }
                }
                System.out.println("해당 ID는 존재하지 않습니다. 다시 시도해주세요");
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
        Outter:while (true) {
            System.out.print("ID : ");
            aID = sc.nextLine();
            if (aID.equals(admin.getId())) {
                inner:while (true) {
                System.out.print("PW : ");
                aPW = sc.nextLine();
                    if(aPW == admin.getPw()) {
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

