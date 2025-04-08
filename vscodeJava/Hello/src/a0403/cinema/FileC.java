package a0403.cinema;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

public class FileC {
    CinemaManager cm = new CinemaManager();

    public void ticket2File(Map<String, Movie> reservationMap, String name) {
        try {
            // File file = new File("c:\\fullStack\\cinema\\ticket.txt");
            File file = new File("e:\\cinema\\ticket.txt");
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file));
            // BufferedWriter를 사용하여 file에 데이터를 쓸 준비
            // FileWriter는 기본적으로 기존 파일을 덮어쓴다.
            if(file.isFile() && file.canWrite()) {
                // .canWrite() - 쓰기 권한이 있는지 true / false
                bufferedWriter.write(cm.ticket(reservationMap, name)); // 티켓정보를 파일에 쓰기
                bufferedWriter.flush(); // 버퍼에 있는 내용을 파일에 저장
                System.out.println("티켓 출력 완료");
                System.out.println();
                bufferedWriter.close();
            }
        } catch (IOException e) {
            System.out.println("티켓 출력 실패");
            System.out.println();
        }    
    }
    
    public void addMovie() {
        try {
            // File file = new File("c:\\fullstack\\cinema\\movies.txt");
            // File addfile = new File("c:\\fullstack\\cinema\\addMovies.txt");
            File file = new File("e:\\cinema\\movies.txt");
            File addfile = new File("e:\\cinema\\addMovies.txt");
            BufferedReader br = new BufferedReader(new FileReader(addfile));
            BufferedWriter bw = new BufferedWriter(new FileWriter(file, true));
            // BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileReader(file)));
            String line;
            System.out.println("========================================");
            while ((line = br.readLine()) != null) {
                System.out.println(line);
                String[] m = line.split("/");
                CinemaManager.getMovies().add(new Movie(m[0], m[1], Integer.parseInt(m[2]), Boolean.parseBoolean(m[3])));
                bw.write("\n" + line);
                bw.flush();
            }
            System.out.println("========================================");
            br.close();
            bw.close();
        } catch (FileNotFoundException e) {
            System.out.println("파일을 찾을 수 없습니다.");
        } catch (IOException e) {
            System.out.println("파일 읽기 실패");
        }
    }

    public void defaultMovie() {
        try {
            // File file = new File("c:\\fullstack\\cinema\\movies.txt");
            File file = new File("e:\\cinema\\movies.txt");

            BufferedReader br = new BufferedReader(new FileReader(file));
            String line;
            System.out.println("========================================");
    
            while ((line = br.readLine()) != null) {
                // System.out.println(line);
                String[] m = line.split("/");
                CinemaManager.getMovies().add(new Movie(m[0], m[1], Integer.parseInt(m[2]), Boolean.parseBoolean(m[3])));
            }
            br.close();
    
        } catch (FileNotFoundException e) {
            System.out.println("파일을 찾을 수 없습니다.");
        } catch (IOException e) {
            System.out.println("파일을 읽을 수 없습니다 ");
        }
    }
    
}    
