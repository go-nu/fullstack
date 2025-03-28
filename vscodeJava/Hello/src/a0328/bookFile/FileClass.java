package a0328.bookFile;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class FileClass {
    private File file;
    private String dir;
    private String fileName;

    public FileClass() {
        file = new File("e:\\");
    }

    public FileClass(String dir, String fileName) {
        file = new File("e:\\" + dir + "\\" + fileName + ".txt");
        this.dir = "e:\\"+ dir;
        this.fileName = fileName + ".txt";
    }

    public void create() throws IOException {
        if(file.exists()) {
            file.delete();
            file.createNewFile();
        } else {
            file = new File(dir);
            file.mkdir();
            file = new File(dir + "\\" + fileName);
            file.createNewFile();
        }
    }

    public void write(String str) throws IOException {
        try {
            FileWriter fw = new FileWriter(file);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(str);
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void read() throws IOException{
        if(file.exists()) {
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);
            String str;
            while((str = br.readLine()) != null){
                System.out.println(str);
            }
            br.close();
        } else {
            System.out.println("읽을 파일이 없습니다.");
        }
    }
    // public void read() throws IOException{
    //     if(file.exists()) {
    //         FileReader fr = new FileReader(file);
    //         BufferedReader br = new BufferedReader(fr);
    //         String str = br.readLine();
    //         while (str != null) {
    //             System.out.println(str);
    //         }
    //         br.close();
    //     } else {
    //         System.out.println("읽을 파일이 없습니다.");
    //     }
    // }
    
}
