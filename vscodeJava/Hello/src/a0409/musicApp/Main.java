package a0409.musicApp;

public class Main {
    public static void main(String[] args) {
        System.out.println("MusicApp 실행");
        Menu menu = LoginMenu.getInstance();

        while (menu != null) {
            menu.print();
            menu = menu.next();
        }
        System.out.println("MusicApp 종료");
    }
}

interface Menu{
    void print();
    Menu next();    
}