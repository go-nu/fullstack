package a0328.file;

import java.io.FileOutputStream;
import java.io.IOException;

public class File2 {
    public static void main(String[] args) throws IOException {
        FileOutputStream output = new FileOutputStream("e:/out.txt");   
        for(int i = 1; i < 11; i++) {
            String data = i + "번째 줄입니다.\r\n";
            // \r\n 줄바꿈 유니스 \n
            output.write(data.getBytes());
        }
        
        output.close();



    }
}
