package a0328.file2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class FileClass {
    private File file; // 자바에서 제공하는 File 객체
    private String dir; // 디렉토리(폴더)
    private String fileName; // 파일명

    public FileClass() {
        file = new File("e:\\");
    }

    public FileClass(String dir, String fileName) {
        file = new File("e:\\" + dir + "\\" + fileName + ".txt");
        this.dir = "e:\\"+ dir;
        this.fileName = fileName + ".txt";
    }

    private boolean check(File f) {
        if(f.exists()) {
            return true;
        }
        return false;
    }

    public void create() throws IOException {
        boolean exist = check(file);
        if (exist) {
            file.delete(); // 파일이 있으면 지운다. (File class에서 제공하는 삭제 메서드)
            file.createNewFile(); // 새 파일 생성
        } else {
            file = new File(dir); // e:\dir -> e:\student
            file.mkdirs(); // 디렉토리 생성
            file = new File(dir + "\\" + fileName); // d:/student/student_Grade.txt
            file.createNewFile();
        }
    }

    public void write(String str) throws IOException {
        FileWriter fw = new FileWriter(file); // 파일 쓰기를 위한 FileWriter 생성
        PrintWriter pw = new PrintWriter(fw); // 파일에 데이터를 출력하기 편리한 메서드(println, printf 사용 가능)
        pw.println(str);
        fw.close();
    }

    public void read() throws IOException {
        boolean exist = check(file);
        if(exist) {
            FileReader fr = new FileReader(file);
            BufferedReader bw = new BufferedReader(fr);
            String str;
            while((str = bw.readLine()) != null){
                System.out.println(str);
            }
            bw.close();
        } else {
            System.out.println("읽을 파일이 없습니다.");
        }
    }


}
