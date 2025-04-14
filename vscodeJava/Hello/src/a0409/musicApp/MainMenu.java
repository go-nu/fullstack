package a0409.musicApp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

class MainMenu extends AbstractMenu {
    File file = new File(".\\src\\a0409\\musicApp\\" + user + ".txt");
    ArrayList<Song> uPlayList = playListMap.get(user);
    private static final MainMenu instance = new MainMenu(null);
    public static MainMenu getInstance() {
        return instance;
    }

    private static final String Main_Menu_Text = 
    "===================================\n" + 
    "메인 메뉴입니다. 메뉴를 선택해주세요\n" + 
    "1.전체 노래\t2. 노래 검색\t3.내 플레이리스트 보기\t4. 플레이리스트 추가\t5. 플레이리스트 삭제\t6. 노래추천\t7.노래공유\t9.이전메뉴\t0.종료\n" + 
    "===================================\n" +
    "선택 >>";

    public MainMenu(Menu prevMenu) {
        super(Main_Menu_Text, prevMenu);
        
        if (uPlayList == null || uPlayList.isEmpty()) {
            uPlayList = new ArrayList<>();
        }
    }

    @Override
    public Menu next() {
        System.out.println("\n" + user + "님 안녕하세요.");
        int ms = sc.nextInt();
        sc.nextLine();

        switch (ms) {
            case 1:
                showAll();
                return this;
            case 2:
                searchSong();
                return this;
            case 3:
                showPL();
                return this;
            case 4:
                addPL();
                return this;
            case 5:
                delPL();
                return this;
            case 6:
                recommendSong();
                return this;
            case 7:
                sharePL();
                return this;
            case 9:
                return prevMenu;
            case 0:
                return null;
            default:
                return this;
        }
    }

    public void create() throws IOException {
        if(file.exists()) {
            file.delete();
            file.createNewFile();
        } else {
            file = new File(".\\src\\a0409\\musicApp\\" + user + ".txt");
            file.createNewFile();
        }
    }

    public void sharePL() {
        try {
            create();
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));
            if(file.isFile() && file.canWrite()) {
                String str = "";
                for (int i = 0; i < uPlayList.size(); i++) {
                    str += uPlayList.get(i).getArtist() + "," + uPlayList.get(i).getTitle() + "," + uPlayList.get(i).getGenre() + "\n";
                }
                bw.write(str);
                bw.flush();
                System.out.println("공유할 플레이리스트 파일 생성 완료");
                System.out.println();
                bw.close();
            }
        } catch (IOException e) {
            System.out.println("파일 생성 실패");
        }
    }

    private void searchSong() {
        System.out.println("-----------------------------------");
        System.out.println("노래 찾기[노래 제목 및 아티스트 검색]");
        System.out.print("검색어를 입력해주세요: ");
        String sWord = sc.nextLine();
    
        boolean found = false;
        for (int i = 0; i < aPlayList.size(); i++) {
            Song song = aPlayList.get(i);
            if (song.getTitle().equalsIgnoreCase(sWord) || song.getArtist().equalsIgnoreCase(sWord)) {
                System.out.println((i + 1) + ". " + song);
                found = true;
            }
        }
    
        if (!found) {
            System.out.println("검색어와 일치하는 노래가 없습니다.");
        }
    
        System.out.println("-----------------------------------");
    }
    


    private void showPL() {
        ArrayList<Song> uPlayList = playListMap.get(user);
        System.out.println("-----------------------------------");
        System.out.println("내 플레이리스트");
        if (uPlayList.size() == 0 || uPlayList.isEmpty()) {
            System.out.println("플레이리스트가 비어있습니다.");
        } else {
            for (int i = 0; i < uPlayList.size(); i++) {
                System.out.println((i+1) + ". " + uPlayList.get(i));
            }
        }
        System.out.println("-----------------------------------");
    }

    private void addPL() {
        ArrayList<Song> uPlayList = playListMap.get(user);
        System.out.println("-----------------------------------");
        System.out.println("내 플레이리스트에 노래 추가하기");
        showAll();
        System.out.print("어느 노래를 플레이리스트에 추가하시겠습니까? : ");
        int sNum = sc.nextInt();
        sc.nextLine();
        uPlayList.add(aPlayList.get(sNum-1));
        System.out.println("추가 완료.");
        playListMap.put(user, uPlayList);
    }

    private void delPL() {
        ArrayList<Song> uPlayList = playListMap.get(user);
        System.out.println("플레이리스트에서 노래 삭제");
        showPL();
        System.out.print("어느 노래를 플레이리스트에서 삭제하시겠습니까? : ");
        int sNum = sc.nextInt();    
        sc.nextLine();
        uPlayList.remove(uPlayList.get(sNum-1));
        System.out.println("삭제 완료.");
        playListMap.put(user, uPlayList);
    }

    private void recommendSong() {
        System.out.println("1. 많은 아티스트 / 2. 비슷한 장르");
        int select = sc.nextInt();
        sc.nextLine();
        switch (select) {
            case 1:
                recommendArtist();
                break;
            case 2:
                recommendGenre();
                break;
        
            default:
                break;
        }
    }

    private void recommendGenre() {
        ArrayList<Song> uPlayList = playListMap.get(user);
        int count = 1;
        System.out.println("-----------------------------------");
        System.out.println("장르 추천(장르)");

        Map<String, Integer> genreCount = new HashMap<>();
        for (Song song : uPlayList) {
            String genre = song.getGenre();
            genreCount.put(genre, genreCount.getOrDefault(genre, 0) + 1);
        }

        String mostGenre = null;
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : genreCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                mostGenre = entry.getKey();
                maxCount = entry.getValue();
            }
        }

        System.out.println("추천 장르: " + mostGenre);
        System.out.println("해당 장르의 추천 곡:");
        for(int i = 0; i < aPlayList.size(); i++) {
            if(mostGenre.equalsIgnoreCase(aPlayList.get(i).getGenre())) {
                System.out.println(count + ". " + aPlayList.get(i));
                count++;
            }
        }
        System.out.println("-----------------------------------");
    }

    private void recommendArtist() {
        ArrayList<Song> uPlayList = playListMap.get(user);
        int count = 1;
        System.out.println("-----------------------------------");
        System.out.println("장르 추천(아티스트)");

        Map<String, Integer> artistCount = new HashMap<>();
        for (Song song : uPlayList) {
            String artist = song.getArtist();
            artistCount.put(artist, artistCount.getOrDefault(artist, 0) + 1);
        }

        String mostArtist = null;
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : artistCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                mostArtist = entry.getKey();
                maxCount = entry.getValue();
            }
        }

        System.out.println("추천 아티스트: " + mostArtist);
        System.out.println("해당 아티스트의 추천 곡:");
        for(int i = 0; i < aPlayList.size(); i++) {
            if(mostArtist.equalsIgnoreCase(aPlayList.get(i).getArtist())) {
                System.out.println(count + ". " + aPlayList.get(i));
                count++;
            }
        }
        System.out.println("-----------------------------------");
    }

}
