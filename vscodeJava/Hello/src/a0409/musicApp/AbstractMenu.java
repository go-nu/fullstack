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

    // 얘네들이 메뉴마다 개별로 존재함, 공유가 안됨
    protected static ArrayList<User> accounts = new ArrayList<>(); // 사용자 목록
    protected static ArrayList<Song> aPlayList = new ArrayList<>(); // 전체 플레이리스트
    protected static Map<String, ArrayList<Song>> playListMap = new HashMap<>(); // 사용자 id, 사용자의 플레이리스트

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
            playListMap.put(user, uPlayList); // 새로 만든 리스트 등록
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
