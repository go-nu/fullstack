package a0327.package2;

import a0327.package1.Parent;

public class Child extends Parent {
    public void display() {
        System.out.println("자식 클래스에 접근: " + message);
        showMessage();
    }
}
