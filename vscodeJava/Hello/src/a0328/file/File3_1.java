package a0328.file;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

public class File3_1 {
    public static void main(String[] args) throws IOException {
        // 폴더 경로
        String folderPath = "e:/abc";
        File folder = new File(folderPath);
        
        // 폴더가 존재하지 않으면 생성
        if(!folder.exists()) {
            if(folder.mkdirs()) { // 폴더생성
                System.out.println("폴더 생성됨: " + folderPath);
            } else {
                System.out.println("폴더 생성 실패");
            }
        }

        PrintWriter pw = new PrintWriter(folderPath + "/out.txt");
        for(int i = 1; i < 11; i++) {
            String data = i + "번째 줄입니다.";
            // println 메서드로 줄바꿈
            pw.println(data);
        }
        
        pw.close();
    }
}
