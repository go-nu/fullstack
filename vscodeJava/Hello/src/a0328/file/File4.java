package a0328.file;

import java.io.FileWriter;
import java.io.IOException;

public class File4 {
    public static void main(String[] args) throws IOException {
        FileWriter fw = new FileWriter("e:/out.txt");
        for(int i = 1; i < 11; i++) {
            String data = i + "번째 줄입니다.\r\n";
            // println 메서드로 줄바꿈
            fw.write(data);
        }
        
        fw.close();

        // 파일을 추가모드로 열기
        FileWriter fw2 = new FileWriter("e:/out.txt", true); // true : 이어쓰기
        for(int i = 11; i < 21; i++) {
            String data = i + "번째 줄입니다.\r\n";
            // println 메서드로 줄바꿈
            fw2.write(data);
        }
        
        fw2.close();
    }
}
