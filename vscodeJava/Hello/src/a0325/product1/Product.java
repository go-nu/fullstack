package a0325.product1;

public class Product {
    private int id;
    private String name;
    private double price;

    @Override
    public String toString() {
        return "Product [id= " + id + ", name= " + name + ", price= $" + price + "]";
    }
    
    public Product(int id, String name, double price) {
        this.id = id;
        this.name = name;
        this.price = price;
    }

    public int getId() {
        return id;
    }
    public void setId(int id) {
        this.id = id;
    }
    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public double getPrice() {
        return price;
    }
    public void setPrice(double price) {
        this.price = price;
    }

}
// getter setter 만드는 이유 -> 변수가 private이기 떄문에 직접 접근을 할 수 없어서 getter와 setter를 통해 접근
// 생성자 만드는 이유 -> 객체를 생성하며 변수를 초기화 하기 위함
// toString() 만드는 이유 -> 원래는 주소를 가르키는데 이를 문자열로 내용을 풀어쓰기 위함