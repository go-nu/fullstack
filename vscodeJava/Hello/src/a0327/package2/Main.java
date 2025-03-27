package a0327.package2;

import a0327.package1.Parent;

public class Main {
    public static void main(String[] args) {
        Child child = new Child();
        child.display(); // 상속받은 클래스 내부에서 접근 가능

        Parent parent = new Parent();
        // System.out.println(parent.message); protected는 다른 패키지에서 직접 접근 불가
        // parent.showMessage(); 상속 없이 직접 호출 불가
    }
}

// 1. public(공개)
// 어디서든 접근 가능(제한 없음)
// 예: 라이브러리, api, 유틸리티 클래스의 메서드 등
// 2. protected(상속 관계에서만 접근 가능)
// 같은 패키지 내에서는 자유롭게 접근 가능
// 다른 패키지에서는 상속 관계에서만 접근 가능
// 3. default(패키지전용)
// 같은 패키지 내에서만 접근가능
// 다른 패키지에서는 접근 불가
// 4. private(비공개)
// 같은 클래스 내부에서만 접근 가능
// 외부에서 직접 접근할 수 없음
// 보통 클래스 내부의 필드(멤버 변수)를 보호나는데 사용
// getter setter를 통한 접근