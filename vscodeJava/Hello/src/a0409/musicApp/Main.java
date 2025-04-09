package a0409.musicApp;

public class Main {
        public static void main(String[] args) {
            LoginMenu lm = new LoginMenu();
            MainMenu m2 = new MainMenu();
            UserMenu um = new UserMenu();
            AdminMenu am = new AdminMenu();



            System.out.println("MusicApp 실행");
            lm.loginMenu();
        }
}
