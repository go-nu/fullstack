package a0327.class1;

public class Math1 {
    public static void main(String[] args) {
        // Math.ceil() 올림
        System.out.println(Math.ceil(10.0)); // 10
        System.out.println(Math.ceil(10.1)); // 11
        // Math.floor() 내림
        System.out.println(Math.floor(10.0));
        System.out.println(Math.floor(10.9));
        // Math.round() 반올림
        System.out.println(Math.round(10.0));
        System.out.println(Math.round(10.4));
        System.out.println(Math.round(10.5));
        
        System.out.println(Math.max(3.14, 3.14159));
        System.out.println(Math.min(3.14, 3.14159));

        System.out.println(Math.max(-10, -11));
        System.out.println(Math.min(-10, -11));

        System.out.println((int)(Math.random() * 100));

    }
}
