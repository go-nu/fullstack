package a0321;

public class HeroTest {
    public static void main(String[] args) {
        Hero thor = new Hero("토르", 150);
        Hero thanos = new Hero("타노스", 160);

        thor.punch(thanos);
        thor.punch(thanos);
        thanos.punch(thor);
    }
}
class Hero {
    String name; // 이름
    int hp; // 체력
    Hero(String n, int h) {
        name = n;
        hp = h;
    }

    void punch(Hero enemy) {
        System.out.printf("[%s]의 펀치! ", name);
        enemy.hp -= 10;
        System.out.printf("-> %s의 체력 : %d\n", enemy.name, enemy.hp);

    }
}