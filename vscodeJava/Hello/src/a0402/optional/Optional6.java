package a0402.optional;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class Optional6 {
    private static final Map<Integer, String> userDatabase = new HashMap<>();
    // 새로운 객체 할당 불가능하게 할 때, final
    static {
        userDatabase.put(1, "Alice");
        userDatabase.put(2, "Bob");
        userDatabase.put(3, "Charlie");
    }
    // static 블럭 - 클래스가 최초 로드될 때 한번만 실행
    // 정적변수(static) 초기화 사용, 객체 생성하지 않아도 실행.

    public static void main(String[] args) {
        // userDatabase = new HashMap<>(); -> 사용 불가

        // 존재하는 id 조회
        Optional<String> user1 = findById(2);
        System.out.println("User with Id 2 : " + user1.orElse("User not found"));

        Optional<String> user2 = findById(5);
        System.out.println("User with Id 5 : " + user2.orElse("User not found"));

        System.out.println("User with Id 3(orElse) : " + findById(3).orElse(getDefaultUser()));
        System.out.println("User with Id 3(orElse) : " + findById(3).orElseGet(() -> getDefaultUser()));
        // orElse(getDefaultUser() 항상 getDefaultUser 호출)
        // orElseGet(() -> getDefaultUser()는 비어 있을 때(null)만 호출);
    }

    private static String getDefaultUser() {
        System.out.println("getDefaultUser() 호출됨.");
        return "Default User";
    }

    private static Optional<String> findById(int id) {
        return Optional.ofNullable(userDatabase.get(id));
    }


}
