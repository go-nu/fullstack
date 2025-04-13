package a0409.musicApp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

// 상속 클래스
// 이 클래스를 상속하는 다른 클래스에서는 모든 elements에 대하여 접근할 수 있음
abstract class AbstractMenu implements Menu {
    protected String menuText;
    protected Menu prevMenu;
    protected String user;

    protected ArrayList<User> accounts = new ArrayList<>(); // 사용자 목록
    protected ArrayList<Song> aPlayList = new ArrayList<>(); // 전체 플레이리스트
    protected Map<String, ArrayList<Song>> playListMap = new HashMap<>(); // 사용자 id, 사용자의 플레이리스트

    protected static final Scanner sc = new Scanner(System.in);

    public AbstractMenu(String menuText, Menu prevMenu) {
        this.menuText = menuText;
        this.prevMenu = prevMenu;
    }

    public void print() {
        System.out.println("\n" + menuText);
    }

    public void setPrevMenu(Menu prevMenu) {
        this.prevMenu = prevMenu;
    }

    public void setUser(String user) {
        this.user = user;
        ArrayList<Song> uPlayList = playListMap.get(user);
        if (uPlayList == null) {
            uPlayList = new ArrayList<>();
            playListMap.put(user, uPlayList);
        }

    }

    public void showAll() {
        System.out.println("*  *  *  *  *  *  *  *  *  *");
        System.out.println("전체 노래 목록");
        for(int i = 0; i < aPlayList.size(); i++) {
            System.out.println((i+1) + ". " + aPlayList.get(i) + "\n");
        }
        System.out.println("*  *  *  *  *  *  *  *  *  *");
    }

}
