package a0328.file;

import java.io.FileInputStream;
import java.io.IOException;

public class File6 {
    public static void main(String[] args) throws IOException{
        byte[] b = new byte[1024];
        FileInputStream input = new FileInputStream("e://out.txt");
        input.read(b);
        System.out.println(new String(b)); // byte를 문자열로 변환
        input.close();
    }
}
