package a0328.file1;

import java.io.File;

public class FileHave {
    public static void main(String[] args) {
         File dir = new File("e:\\abc"); // 특정 폴더 지정

        if (dir.exists() && dir.isDirectory()) { // 폴더인지 확인
            String[] fileList = dir.list(); // 폴더 내 파일 목록 가져오기
            System.out.println("폴더 내 파일 목록:");
            for (String fileName : fileList) {
                System.out.println(fileName);
            }
        } else {
            System.out.println("폴더가 존재하지 않습니다.");
        }
    }
}
/*
createNewFile()   새 파일 생성
delete()   파일/폴더 삭제
mkdir()   폴더 생성
renameTo(File dest)   파일/폴더 이름 변경 또는 이동
exists()   파일/폴더 존재 여부 확인
isFile()   파일인지 확인
isDirectory()   폴더인지 확인
length()   파일 크기(바이트) 반환
list()   폴더 내 파일 목록 반환
 */