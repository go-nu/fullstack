package com.shop.service;

import lombok.extern.java.Log;
import org.springframework.stereotype.Service;
import java.io.File;
import java.io.FileOutputStream;
import java.util.UUID;

@Service
@Log
public class FileService {
    // uploadPath : 저장경로, itemImgLocation에서 넘어옴, 예)C:/shop/item
    public String uploadFile(String uploadPath, String originalFileName, byte[] fileData) throws Exception{
        UUID uuid = UUID.randomUUID(); // 고유ID(랜덤 파일명) 생성
        String extension = originalFileName.substring(originalFileName.lastIndexOf(".")); // 확장자 추출
        String savedFileName = uuid.toString() + extension; // 고유ID에 확장자를 합쳐서 업로드
        String fileUploadFullUrl = uploadPath + "/" + savedFileName; // C:/shop/item/sadnjasdfkj.jpg
        FileOutputStream fos = new FileOutputStream(fileUploadFullUrl);
        fos.write(fileData);
        // 바이트 배열로 전달받은 데이터를 해당 경로에 파일로 저장
        fos.close();
        return savedFileName;
    }

/*
    public String uploadFile(String uploadPath, String originalFileName, byte[] fileData) throws Exception {
        UUID uuid = UUID.randomUUID();
        String extension = originalFileName.substring(originalFileName.lastIndexOf("."));
        String savedFileName = uuid.toString() + extension;

        Path uploadDir = Paths.get(uploadPath);
        Files.createDirectories(uploadDir); // 디렉토리 없으면 생성

        Path filePath = uploadDir.resolve(savedFileName); // 경로 조합 (OS 호환)
        Files.write(filePath, fileData); // 파일 저장

        return savedFileName;
    }
*/


    public void deleteFile(String filePath) throws Exception{
        File deleteFile = new File(filePath);
        if(deleteFile.exists()) {
            deleteFile.delete();
            log.info("파일을 삭제하였습니다.");
        } else {
            log.info("파일이 존재하지 않습니다.");
        }
    }

}