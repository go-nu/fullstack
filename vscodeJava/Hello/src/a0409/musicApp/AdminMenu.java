package a0409.musicApp;

class AdminMenu extends AbstractMenu{
    FileC fc = new FileC();
    private static final AdminMenu instance = new AdminMenu(null);
    public static AdminMenu getInstance() {
        return instance;
    }

    private static final String Admin_Menu_Text = 
    "===================================\n" + 
    "관리자 화면입니다. 메뉴를 선택해주세요\n" + 
    "1. 노래 추가\t2. 노래 제거\t3. 노래 정보 변경\t4. 사용자 정보 출력\t0.종료\n" + 
    "===================================\n" +
    "선택 >>";

    public AdminMenu(Menu prevMenu) {
        super(Admin_Menu_Text, prevMenu);
    }

    @Override
    public Menu next() {
        int as = sc.nextInt();
        sc.nextLine();

        switch (as) {
            case 1:
                addNewSong();
                return this;
            case 2:
                delSong();
                return this;
            case 3:
                updateSong();
                return this;
            case 4:
                showUser();
                return this;
            case 0:
                return null;
            default:
                return this;
        }
    }

    private void addNewSong() {
        System.out.println("-----------------------------------");
        System.out.println("노래 추가하기");
        fc.add();
    }

    private void delSong() {
        int dNum;
        System.out.println("-----------------------------------");
        System.out.println("노래 제거하기");
        System.out.println("*  *  *  *  *  *  *  *  *  *");
        System.out.println("전체 노래 목록");
        for(int i = 0; i < aPlayList.size(); i++) {
            System.out.println((i+1) + ". " + aPlayList.get(i) + "\n");
        }
        System.out.println("*  *  *  *  *  *  *  *  *  *");
        Outter:while (true) {
            System.out.print("제거할 노래 번호 선택 >> ");
            dNum = sc.nextInt();
            sc.nextLine();
            if (dNum < 1 || dNum > aPlayList.size()) {
                System.out.println("잘못된 입력입니다.");
                continue;
            } else {
                break Outter;
            }
        }

        fc.del(dNum);
    }

    private void updateSong() {
        System.out.println("-----------------------------------");
        System.out.println("노래 정보 수정하기");
        fc.update();
    }

    private void showUser() {
        System.out.println("-----------------------------------");
        System.out.println("사용자 모두 보기");
    }

}
