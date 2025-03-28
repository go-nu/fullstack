package a0328.file;

import java.io.IOException;
import java.io.PrintWriter;

public class File3 {
    public static void main(String[] args) throws IOException {
        PrintWriter pw = new PrintWriter("e:/out.txt");
        for(int i = 1; i < 11; i++) {
            String data = i + "번째 줄입니다.";
            // println 메서드로 줄바꿈
            pw.println(data);
        }
        
        pw.close();
    }
}
