package board.kkw.service;

import board.kkw.domain.Board;
import board.kkw.domain.Member;
import board.kkw.dto.BoardDTO;
import board.kkw.repository.BoardRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.UUID;

@Service
public class BoardService {

    @Autowired
    private BoardRepository boardRepository;
    @Value("${upload.dir}")
    private String uploadDir;

    public void create(BoardDTO dto, Member writer) throws IOException {
        Board board = new Board();
        board.setTitle(dto.getTitle());
        board.setWriter(writer.getUserId());
        board.setContent(dto.getContent());
        if (!dto.getImg().isEmpty()) {
            // 파일 이름 생성 (UUID를 이용하여 랜덤한 파일명 생성)
            String filename = UUID.randomUUID() + "_" + dto.getImg().getOriginalFilename();
            // 파일 저장 경로 생성 (uploadDir은 application.properties에서 경로 설정)
            Path path = Paths.get(uploadDir, filename);
            // 디렉토리가 없으면 생성
            Files.createDirectories(path.getParent());
            // 실제 이미지 저장
            Files.copy(dto.getImg().getInputStream(), path);
            // DB에 저장할 이미지 경로 설정
            board.setImgPath("/upload/" + filename);
        }
        boardRepository.save(board);
    }

    public List<Board> list() {
        return boardRepository.findAll();
    }

    public Board findById(Long num) {
        return boardRepository.findById(num).orElseThrow();
    }

    public void update(Long num, BoardDTO dto) throws IOException  {
        Board board = findById(num);
        board.setTitle(dto.getTitle());
        board.setContent(dto.getContent());
        if (!dto.getImg().isEmpty()) {
            // 파일 이름 생성 (UUID를 이용하여 랜덤한 파일명 생성)
            String filename = UUID.randomUUID() + "_" + dto.getImg().getOriginalFilename();
            // 파일 저장 경로 생성 (uploadDir은 application.properties에서 경로 설정)
            Path path = Paths.get(uploadDir, filename);
            // 디렉토리가 없으면 생성
            Files.createDirectories(path.getParent());
            // 실제 이미지 저장
            Files.copy(dto.getImg().getInputStream(), path);
            // DB에 저장할 이미지 경로 설정
            board.setImgPath("/upload/" + filename);
        }
        boardRepository.save(board);
    }

    public void delete(Board board) {
        boardRepository.delete(board);
    }

}
