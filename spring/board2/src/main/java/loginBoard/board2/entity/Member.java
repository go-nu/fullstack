package loginBoard.board2.entity;

import jakarta.persistence.*;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Member {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String username;
    private String password;

    // 회원이 작성한 글
    @OneToMany(mappedBy = "writer", cascade = CascadeType.REMOVE, orphanRemoval = true)
    // Member를 삭제할 때, 해당 Member가 작성한 모든 Board 같이 삭제
    private List<Board> boards = new ArrayList<>();
    // user는 한명이고, 여러 board를 가질 수 있다.
    // board는 하나의 user(writer)만을 가진다.
    // board가 연관관계의 주인, user는 mappedBy로 관계만 나타냄

    @Override
    public String toString() {
        return "Member{" +
                "id=" + id +
                ", username='" + username + '\'' +
                ", password='" + password + '\'' +
                '}';
    }
}
