package a0324;

public class SingletonMain {
    public static void main(String[] args) {
        
        Singleton s1 = Singleton.getInstance();
        Singleton s2 = Singleton.getInstance();
        Singleton s3 = Singleton.getInstance();

        // Singleton s4 = new Singleton();

        // 객체를 한번만 만들기 위해서?
        // 싱글톤 패턴은 하나의 인스턴스만 유지하여 메모리 절약과 성능 최적화

    }
}
