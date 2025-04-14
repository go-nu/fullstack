package a0409.musicApp;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

class AdminMenu extends AbstractMenu{
    // File file = new File("C:\\Users\\admin\\Desktop\\webHome\\vscodeJava\\Hello\\src\\a0409\\musicApp\\" + user + ".txt");
    File file = new File(".\\src\\a0409\\musicApp\\" + user + ".txt");
    private static final AdminMenu instance = new AdminMenu(null);
    public static AdminMenu getInstance() {
        return instance;
    }

    private static final String Admin_Menu_Text = 
    "===================================\n" + 
    "관리자 화면입니다. 메뉴를 선택해주세요\n" + 
    "1. 노래 추가\t2. 노래 제거\t3. 사용자 정보 출력\t4.전체보기\t9.이전메뉴\t0.종료\n" + 
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
                showUser();
                return this;
            case 4:
                showAll();
                return this;
            case 9:
                return prevMenu;
            case 0:
                return null;
            default:
                return this;
        }
    }

    private void addNewSong() {
        System.out.println("-----------------------------------");
        System.out.println("노래 추가하기");
        try {
            // File file = new File("C:\\Users\\admin\\Desktop\\webHome\\vscodeJava\\Hello\\src\\a0409\\musicApp\\defaultPlayList.txt");
            // File addfile = new File("C:\\Users\\admin\\Desktop\\webHome\\vscodeJava\\Hello\\src\\a0409\\musicApp\\addPlayList.txt");
            File file = new File(".\\src\\a0409\\musicApp\\defaultPlayList.txt");
            File addfile = new File(".\\src\\a0409\\musicApp\\addPlayList.txt");
            BufferedReader br = new BufferedReader(new FileReader(addfile));
            BufferedWriter bw = new BufferedWriter(new FileWriter(file, true));
            String line;
            System.out.println("*  *  *  *  *  *  *  *  *  *");
            while ((line = br.readLine()) != null) {
                System.out.println(line);
                String[] s = line.split(",");
                aPlayList.add(new Song(s[0], s[1], s[2]));
                bw.write("\n" + line);
                bw.flush();
            }
            System.out.println("*  *  *  *  *  *  *  *  *  *");
            br.close();
            bw.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    private void delSong() {
        int dNum;
        System.out.println("-----------------------------------");
        System.out.println("노래 제거하기");
        showAll();
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

        int lineIndex = 1;
        try {
            // File file = new File("C:\\Users\\admin\\Desktop\\webHome\\vscodeJava\\Hello\\src\\a0409\\musicApp\\defaultPlayList.txt");
            File file = new File(".\\src\\a0409\\musicApp\\defaultPlayList.txt");
            BufferedReader br = new BufferedReader(new FileReader(file));
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));
            String line;
            System.out.println("*  *  *  *  *  *  *  *  *  *");
            while ((line = br.readLine()) != null) {
                lineIndex++;
                if (dNum == lineIndex) {
                    continue;
                }
                System.out.println(line);
                String[] s = line.split(",");
                aPlayList.add(new Song(s[0], s[1], s[2]));
                bw.write("\n" + line);
                bw.flush();
            }
            System.out.println("*  *  *  *  *  *  *  *  *  *");
            br.close();
            bw.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    private void showUser() {
        System.out.println("-----------------------------------");
        System.out.println("사용자 모두 보기");
        System.out.println("이름[YYYYMMDD] ID : id / PW : pw");
        for (int i = 0; i < accounts.size(); i++) {
            System.out.printf("%s[%d] ID : %s / PW : %s\n",
            accounts.get(i).getName(), accounts.get(i).getBirth(), accounts.get(i).getId(), accounts.get(i).getPw());
        }
    }

}
