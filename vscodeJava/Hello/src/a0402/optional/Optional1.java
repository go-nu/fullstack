package a0402.optional;

import java.util.Optional;

// OPtional : null 값으로 인한 문제 방지, 안전하게 값을 처리
// null이 될 수 있는 객체를 감싸는 wrapper class
public class Optional1 {
    public static void main(String[] args) {
        Optional<String> name = Optional.of("John");
        System.out.println("Name : " + name.get());
        
        // 빈 Optional 생성
        Optional<String> emptyName = Optional.empty();
        System.out.println("Is empty : " + emptyName.isPresent()); // false
        
        // Optional.ofNullable() = null을 허용하는 Optional 생성
        Optional<String> nullAbleName = Optional.ofNullable(null);
        System.out.println("Is empty : " + nullAbleName.isPresent()); // false
        

    }
}
