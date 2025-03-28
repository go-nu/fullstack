package a0328.file1;

import java.io.File;

public class FileInfo {
    public static void main(String[] args) {
        File file = new File("e:\\abc\\out.txt");

        // 파일 정보 출력
        System.out.println("파일 이름: " + file.getName());
        System.out.println("파일 경로: " + file.getPath());
        System.out.println("파일 존재 여부: " + file.exists());
    }
}
