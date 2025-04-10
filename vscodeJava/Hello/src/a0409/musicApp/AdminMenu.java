package a0409.musicApp;

class AdminMenu extends AbstractMenu{
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
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'addNewSong'");
    }

    private void delSong() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'delSong'");
    }

    private void updateSong() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'updateSong'");
    }

    private void showUser() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'showUser'");
    }

}
