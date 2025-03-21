package a0321.Account1;

public class CoffeeTest {
    public static void main(String[] args) {
        Coffee c = new Coffee("아메리카노", 3000);
        System.out.printf("%s(%d) -> ", c.getName(), c.getPrice());
        c.setPrice(c.getPrice() + 500);
        System.out.printf("%s(%d)", c.getName(), c.getPrice());
    }
}
class Coffee {
    private String name;
    private int price;

    public String getName() {
        return name;
    }
    public void setName(String n) {
        this.name = n;
    }
    public int getPrice() {
        return price;
    }
    public void setPrice(int p) {
        this.price = p;
    }

    Coffee(String n, int p) {
        name = n;
        price = p;
    }

}