package a0409.musicApp;

import java.util.ArrayList;
import java.util.List;

class MainMenu extends AbstractMenu {
    private static final MainMenu instance = new MainMenu(null);
    public static MainMenu getInstance() {
        return instance;
    }

    private static final String Main_Menu_Text = 
    "===================================\n" + 
    "메인 메뉴입니다. 메뉴를 선택해주세요\n" + 
    "1. 노래 검색\t2.플레이리스트 보기\t3. 플레이리스트 추가\t4. 플레이리스트 삭제\t5. 노래추천\t0.종료\n" + 
    "===================================\n" +
    "선택 >>";

    public MainMenu(Menu prevMenu) {
        super(Main_Menu_Text, prevMenu);
    }

    @Override
    public Menu next() {
        int ms = sc.nextInt();
        sc.nextLine();

        switch (ms) {
            case 1:
                searchSong();
                return this;
            case 2:
                showPL();
                return this;
            case 3:
                addPL();
                return this;
            case 4:
                delPL();
                return this;
            case 5:
                recommendSong();
                return this;
            case 0:
                return null;
            default:
                return this;
        }
    }

    private void searchSong() {
        System.out.println("-----------------------------------");
        System.out.println("노래 찾기[노래 제목 및 아티스트 검색]");
        for (int i = 0; i < findSong(aPlayList).size(); i++) {
            System.out.println(findSong(aPlayList).get(i));
        }
        System.out.println("-----------------------------------");
    }

    private List<Song> findSong(ArrayList<Song> pL) {
        System.out.print("검색어를 입력해주세요 : ");
        String sWord = sc.nextLine();
        
        List<Song> searchResults = new ArrayList<>();
        
        for (int i = 0; i < pL.size(); i++) {
            String fTitle = pL.get(i).getTitle();
            String fArtist = pL.get(i).getArtist();
            if (sWord.equalsIgnoreCase(fTitle) || sWord.equalsIgnoreCase(fArtist)) {
                searchResults.add(pL.get(i));
            }
        }
        
        if (searchResults.isEmpty()) {
            System.out.println("검색어와 일치하는 결과가 없습니다.");
        }
        
        return searchResults;
    }

    private void showPL() {
        System.out.println("-----------------------------------");
        System.out.println("내 플레이리스트");
        if (uPlayList.size() == 0 || uPlayList.isEmpty()) {
            System.out.println("플레이리스트가 비어있습니다.");
        } else {
            for (int i = 0; i < uPlayList.size(); i++) {
                System.out.println(uPlayList.get(i));
            }
        }
        System.out.println("-----------------------------------");
    }

    private void addPL() {
        System.out.println("-----------------------------------");
        System.out.println("내 플레이리스트에 노래 추가하기");
        System.out.println("*  *  *  *  *  *  *  *  *  *");
        for (int i = 0; i < findSong(aPlayList).size(); i++) {
            System.out.println(findSong(aPlayList).get(i));
        }
        System.out.println("*  *  *  *  *  *  *  *  *  *");
        System.out.print("어느 노래를 플레이리스트에 추가하시겠습니까? : ");
        int sNum = sc.nextInt();
        sc.nextLine();
        uPlayList.add(findSong(aPlayList).get(sNum));
        System.out.println("추가 완료.");
    }

    private void delPL() {
        System.out.println("플레이리스트에서 노래 삭제");
        showPL();
        System.out.println("*  *  *  *  *  *  *  *  *  *");
        for (int i = 0; i < findSong(uPlayList).size(); i++) {
            System.out.println(findSong(uPlayList).get(i));
        }
        System.out.println("*  *  *  *  *  *  *  *  *  *");
        System.out.print("어느 노래를 플레이리스트에서 삭제하시겠습니까? : ");
        int sNum = sc.nextInt();
        sc.nextLine();
        uPlayList.remove(sNum);
        System.out.println("삭제 완료.");
    }

    private void recommendSong() {
        
    }

}
