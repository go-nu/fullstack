package a0324.rpg2;

public class OverridingTest {
    public static void main(String[] args) {
        Archer a = new Archer();
        Archer ma = (Archer)new MasterArcher(); // 업캐스팅, 생략가능
        a.shoot();
        ma.shoot();
    }
}
class Archer{
    void shoot() {
        System.out.println("[아처]의 활 공격이 10만큼 피해를 주었다.");
    }
}
class MasterArcher extends Archer{
    void shoot() {
        System.out.println("[마스터아처]의 활 공격이 30만큼 피해를 주었다.");
    }
}
// 래퍼런스(참조) 변수 ma는 연결 객체를 Archer라 해석하지만, 실제 연결된 객체는 MasterArcher이므로
// 매서드 오버라이딩이 된다. 그 결과, Archer 클래스의 shoot()이 아닌 MasterArcher의 shoot() 메소드가 수행