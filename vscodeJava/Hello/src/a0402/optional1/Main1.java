package a0402.optional1;

import java.util.Optional;

public class Main1 {
    public static void main(String[] args) {
        UserRepository userRepository = new UserRepository();

        // 존재하는 id 검색
        Optional<User> userOptional1 = userRepository.findById(2L);
        userOptional1.ifPresentOrElse(
            user -> System.out.println("찾는 사용자: " + user),
            () -> System.out.println("찾는 사용자가 없습니다.")
        ); // 출력> 찾는 사용자: User{id = 2, name = Bob}

        Optional<User> userOptional2 = userRepository.findById(99L);
        userOptional2.ifPresentOrElse(
            user -> System.out.println("찾는 사용자: " + user),
            () -> System.out.println("찾는 사용자가 없습니다.")
        ); // 출력> 찾는 사용자가 없습니다.

        Optional<User> userOptional3 = userRepository.findById(50L);
        userOptional3.ifPresentOrElse(
            user -> System.out.println("찾는 사용자: " + user),
            () -> System.out.println("찾는 사용자가 없습니다.")
        ); // 출력> 찾는 사용자가 없습니다.

        // 기본값 제공
        User defaultUser = userOptional3.orElse(new User(0L, "DefaultUser"));
        System.out.println("기본 사용자 : " + defaultUser);
    }
}

// findById()
// 아이디로 사용자 리스트를 검색하고, 결과를 Optional<User>로 반환
// Optional.empty()는 사용자가 없을 때 반환.
// Optional 처리방식 >
// ifPresentOrElse(a, b) : 값이 있으면 a 처리, 없으면 대체 동작(b) 수행
// orElse(a) : 값이 없으면 기본값(a) 반환
// orElseThrow : 값이 없으면 예외를 던지기