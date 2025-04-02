package a0402.optional1;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class UserRepository {
    private List<User> users = new ArrayList<>();

    public UserRepository(){
        // 더미 데이터 추가
        users.add(new User(1L, "Alice"));
        users.add(new User(2L, "Bob"));
        users.add(new User(3L, "Charlie"));
   }

    public Optional<User> findById(long id) {
        // for(User user : users) { // users list를 돌면서
        //     if(user.getId() == id) { // 조건에 맞는 user 찾기
        //         return Optional.of(user);
        //     }
        // }
        // return Optional.empty(); // 없으면 빈 Optional 반환

        return users.stream()
                    .filter(u -> (u.getId() == id))
                    .findFirst();
                    // findFirst() 처음 조건을 만족하는 객체를 Optional<T>로 감싸 반환
                    // 없으면 Optional.empty()
    }


}
