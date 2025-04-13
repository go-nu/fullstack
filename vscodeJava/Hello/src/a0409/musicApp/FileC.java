package a0409.musicApp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class FileC {
    private File file;

    private AbstractMenu menu;

    public FileC() {};

    public FileC(AbstractMenu menu) {
        this.menu = menu;
    }

    public void create() throws IOException {
        if(file.exists()) {
            file.delete();
            file.createNewFile();
        } else {
            file = new File(".\\" + menu.user + ".txt");
            file.createNewFile();
        }
    }

    public void sharePL() {
        try {
            create();
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));
            if(file.isFile() && file.canWrite()) {
                bw.write("");
                bw.flush();
                System.out.println("공유할 플레이리스트 파일 생성 완료");
                System.out.println();
                bw.close();
            }

        } catch (IOException e) {
            System.out.println("파일 생성 실패");
            System.out.println();
        }


    }

    public void add() {
        try {
            File file = new File(".\\defaultPlayList.txt");
            File addfile = new File(".\\addPlayList.txt");
            BufferedReader br = new BufferedReader(new FileReader(addfile));
            BufferedWriter bw = new BufferedWriter(new FileWriter(file, true));
            String line;
            System.out.println("*  *  *  *  *  *  *  *  *  *");
            while ((line = br.readLine()) != null) {
                System.out.println(line);
                String[] s = line.split(",");
                menu.aPlayList.add(new Song(s[0], s[1], s[2]));
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

    public void del(int dNum) { // .txt 파일에서 dNum번째 line 제거
        int lineIndex = 1;
        try {
            File file = new File(".\\" + menu.user + ".txt");
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
                menu.aPlayList.add(new Song(s[0], s[1], s[2]));
                bw.write("\n" + line);
                bw.flush();
            }
            System.out.println("*  *  *  *  *  *  *  *  *  *");
            br.close();
            bw.close();
        } catch (Exception e) {
            // TODO: handle exception
        }
    }

    public void update() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'update'");
    }

    public void defaultSong() {
        try {
            File file = new File(".\\defaultPlayList.txt");

            BufferedReader br = new BufferedReader(new FileReader(file));
            String line;
            System.out.println("========================================");
    
            while ((line = br.readLine()) != null) {
                String[] s = line.split(",");
                menu.aPlayList.add(new Song(s[0], s[1], s[2]));
            }
            br.close();
    
        } catch (FileNotFoundException e) {
            System.out.println("파일을 찾을 수 없습니다.");
        } catch (IOException e) {
            System.out.println("파일을 읽을 수 없습니다 ");
        }
    }
}
